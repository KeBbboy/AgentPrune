import ast
import asyncio
import io
import json
import os
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.tools.core import BaseTool, ToolRegistry, ToolResult
from AgentDropout.tools.search.search import GoogleSearchEngine, SearchAPIEngine
from AgentDropout.tools.search.wiki import search_wiki_main

load_dotenv()


DEFAULT_TOOL_CONFIG: Dict[str, Any] = {
    "enabled_tools": [],
    "direct_tool_call": False,
    "web_tool_provider": "tavily",
    "top_k_results": 3,
    "max_search_limit": 3,
    "max_tool_chars": 6000,
    "max_tool_turns": 3,
}


@dataclass
class ToolObservation:
    name: str
    content: str

    @classmethod
    def from_result(cls, name: str, result: ToolResult) -> "ToolObservation":
        if result.success:
            content = result.output
        else:
            content = result.error or result.output or "Tool execution failed."
        return cls(name=name, content=content)


def normalize_enabled_tools(tools: Optional[Iterable[str]]) -> List[str]:
    if tools is None:
        return []
    normalized = []
    for tool in tools:
        tool_name = str(tool).strip()
        if tool_name and tool_name not in normalized:
            normalized.append(tool_name)
    return normalized


def _truncate_text(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


class _BenchmarkToolMixin:
    def __init__(
        self,
        llm_name: str,
        direct_tool_call: bool = False,
        max_tool_chars: int = 6000,
    ) -> None:
        self.llm_name = llm_name
        self.direct_tool_call = direct_tool_call
        self.max_tool_chars = max_tool_chars
        self._llm = None

    @property
    def direct_mode(self) -> bool:
        return bool(self.direct_tool_call)

    def _llm_complete(self, system_prompt: str, user_prompt: str) -> str:
        if self._llm is None:
            self._llm = LLMRegistry.get(self.llm_name)
        response = self._llm.gen(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response if isinstance(response, str) else "\n".join(response)

    def _truncate(self, text: str) -> str:
        text = "" if text is None else str(text)
        if len(text) <= self.max_tool_chars:
            return text
        return text[: self.max_tool_chars] + "\n...[truncated]"


class TextInspectorTool(_BenchmarkToolMixin, BaseTool):
    def __init__(
        self,
        llm_name: str,
        direct_tool_call: bool = False,
        max_tool_chars: int = 6000,
    ) -> None:
        super().__init__(llm_name, direct_tool_call, max_tool_chars)

    @property
    def name(self) -> str:
        return "text_inspector"

    @property
    def description(self) -> str:
        return "Read and optionally analyze attached local text/table/document files"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Inspect a local attachment such as a document, table, PDF, CSV, or text file. "
                    "Use this when the question refers to attached or local files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional file path to inspect. Omit this to inspect provided attachments.",
                        }
                    },
                    "required": [],
                },
            },
        }

    def execute(
        self,
        raw_inputs: Optional[Dict[str, Any]] = None,
        path: Any = None,
        file_path: Any = None,
        file: Any = None,
        filename: Any = None,
        attachment: Any = None,
        **_: Any,
    ) -> ToolResult:
        raw_inputs = raw_inputs or {}
        requested_path = path or file_path or file or filename or attachment
        attachments = raw_inputs.get("attachments") or raw_inputs.get("metadata", {}).get("attachments") or []
        if requested_path:
            attachments = list(requested_path) if isinstance(requested_path, (list, tuple)) else [requested_path]
        try:
            from AgentDropout.tools.reader.readers import GeneralReader
        except ModuleNotFoundError as exc:
            if attachments:
                return ToolResult(success=False, output="", metadata={}, error=f"text_inspector unavailable: missing dependency {exc.name}")
            return ToolResult(success=True, output="", metadata={"attachments": 0})

        reader = GeneralReader()
        sections = []
        for candidate in attachments:
            try:
                if not candidate or not Path(candidate).exists():
                    continue
                content = reader.read(str(raw_inputs.get("question") or raw_inputs.get("task") or ""), str(candidate))
                if content:
                    sections.append(content)
            except Exception as exc:
                sections.append(f"Failed to inspect {candidate}: {exc}")
        output = "\n\n".join(sections)
        return ToolResult(success=True, output=self._truncate(output), metadata={"attachments": len(attachments)})


class WebSearchTool(_BenchmarkToolMixin, BaseTool):
    def __init__(
        self,
        llm_name: str,
        provider: str = "tavily",
        top_k_results: int = 3,
        direct_tool_call: bool = False,
        max_tool_chars: int = 6000,
    ) -> None:
        super().__init__(llm_name, direct_tool_call, max_tool_chars)
        self.provider = provider
        self.top_k_results = max(1, int(top_k_results))
        self._search_engine = None
        self._analysis_cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return f"Search the web for external facts using {self.provider}"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the web for information to help answer questions. Use this when you need facts, "
                    "verification, current information, or source-grounded context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Specific web search query with the key entities and facts to verify.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self,
        query: Optional[str] = None,
        q: Optional[str] = None,
        search_query: Optional[str] = None,
        sub_goal: Optional[str] = None,
        input: Optional[str] = None,
        question: Optional[str] = None,
        raw_inputs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> ToolResult:
        raw_inputs = raw_inputs or {}
        task = str(raw_inputs.get("question") or raw_inputs.get("task") or "").strip()
        resolved_query = str(query or q or search_query or sub_goal or input or question or "").strip()
        if not resolved_query:
            resolved_query = self._build_search_query(task)
        if not resolved_query:
            return ToolResult(success=True, output="", metadata={"query": resolved_query})
        try:
            raw_output = self._search(resolved_query)
            if self.direct_mode:
                output = raw_output
            else:
                output = self._analyze_with_llm(resolved_query, raw_output)
            return ToolResult(
                success=True,
                output=self._truncate(output),
                metadata={"query": resolved_query, "mode": "direct" if self.direct_mode else "sub-agent"},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", metadata={"query": resolved_query}, error=f"Search failed for query '{resolved_query}': {exc}")

    def _build_search_query(self, question: str) -> str:
        if len(question) <= 240:
            return question
        prompt = (
            "Create one concise web search query for answering this benchmark question. "
            "Use key entities and omit answer options unless essential.\n\n"
            f"Question:\n{question}\n\nSearch query:"
        )
        try:
            return self._llm_complete("You write concise web search queries.", prompt).strip().splitlines()[0][:240]
        except Exception:
            return question[:240]

    def _search(self, query: str) -> str:
        provider = self.provider.lower()
        if provider == "wiki":
            queries = [query]
            result = _run_async_from_sync(lambda: search_wiki_main(queries))
            return "\n".join(item for item in result if item)
        if provider == "tavily":
            return self._search_tavily(query)
        if provider == "serper":
            provider = "searchapi"
        if self._search_engine is None:
            if provider == "google":
                self._search_engine = GoogleSearchEngine()
            else:
                self._search_engine = SearchAPIEngine()
        if provider == "google":
            return self._search_engine.search(query, num=self.top_k_results)
        return self._search_engine.search(query, item_num=self.top_k_results)

    def _search_tavily(self, query: str) -> str:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": self.top_k_results,
                "include_answer": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        sections = []
        answer = payload.get("answer")
        if answer:
            sections.append(f"Answer: {answer}")
        for item in payload.get("results", [])[: self.top_k_results]:
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            content = item.get("content") or item.get("raw_content") or ""
            if content:
                sections.append(f"{title}\n{url}\n{content}")
        return "\n\n".join(sections)

    def _analyze_with_llm(self, query: str, raw_output: str) -> str:
        if query in self._analysis_cache:
            return self._analysis_cache[query]
        prompt = (
            "Analyze the web search results and extract only information useful for answering the query. "
            "Be concise, factual, and preserve important names, dates, numbers, and source hints.\n\n"
            f"Query:\n{query}\n\nSearch results:\n{self._truncate(raw_output)}"
        )
        analyzed = self._llm_complete("You are a web-search result analysis sub-agent.", prompt)
        self._analysis_cache[query] = analyzed
        return analyzed


class CodeGeneratorTool(_BenchmarkToolMixin, BaseTool):
    def __init__(
        self,
        llm_name: str,
        direct_tool_call: bool = False,
        max_tool_chars: int = 6000,
        timeout_seconds: int = 10,
    ) -> None:
        super().__init__(llm_name, direct_tool_call, max_tool_chars)
        self.timeout_seconds = max(1, int(timeout_seconds))

    @property
    def name(self) -> str:
        return "code_generator"

    @property
    def description(self) -> str:
        return "Generate and execute Python code for calculations and verification"

    def get_schema(self) -> Dict[str, Any]:
        if self.direct_mode:
            properties = {
                "code": {
                    "type": "string",
                    "description": "Standalone Python code to execute. It must print useful output and contain no markdown fences.",
                }
            }
            required = ["code"]
            description = "Execute Python code to perform calculations, data processing, or verification."
        else:
            properties = {
                "task": {
                    "type": "string",
                    "description": "Clear description of the calculation, transformation, or verification the code should perform.",
                },
                "code": {
                    "type": "string",
                    "description": "Optional pre-written standalone Python code to execute.",
                },
            }
            required = ["task"]
            description = "Generate and execute Python code to perform calculations, data processing, or verification."
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def execute(
        self,
        code: Optional[str] = None,
        python_code: Optional[str] = None,
        program: Optional[str] = None,
        prompt: Optional[str] = None,
        query: Optional[str] = None,
        sub_goal: Optional[str] = None,
        task: Optional[str] = None,
        question: Optional[str] = None,
        input: Optional[str] = None,
        raw_inputs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> ToolResult:
        raw_inputs = raw_inputs or {}
        resolved_code = code or python_code or program
        resolved_task = prompt or query or sub_goal or task or question or input
        benchmark_question = str(raw_inputs.get("question") or raw_inputs.get("task") or "").strip()
        try:
            if not resolved_code:
                if self.direct_mode and resolved_task:
                    resolved_code = str(resolved_task)
                else:
                    generation_task = str(resolved_task or benchmark_question).strip()
                    if not generation_task:
                        return ToolResult(success=True, output="", metadata={})
                    resolved_code = self._generate_code(generation_task)
            resolved_code = _strip_code_fences(str(resolved_code)).strip()
            if not resolved_code or resolved_code.upper().startswith("SKIP"):
                return ToolResult(success=True, output="", metadata={"skipped": True})
            output = _execute_python(resolved_code, timeout_seconds=self.timeout_seconds)
            return ToolResult(
                success=True,
                output=self._truncate(f"Code:\n{resolved_code}\n\nOutput:\n{output}"),
                metadata={"mode": "direct" if self.direct_mode else "sub-agent"},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", metadata={}, error=f"Code helper failed: {exc}")

    def _generate_code(self, task: str) -> str:
        prompt = (
            "Write a short Python calculation, transformation, or verification to help solve the problem. "
            "If code is not useful, return only the word SKIP. "
            "Do not use internet or files unless the prompt explicitly contains a local file path. "
            "Print the useful result.\n\n"
            f"Problem:\n{task}"
        )
        return self._llm_complete(
            "You generate minimal Python snippets for benchmark problem solving. Return only Python code or SKIP.",
            prompt,
        )


class BenchmarkToolRunner:
    """Registry-backed benchmark tool runner using msc-thesis style tools.

    Tools are registered as BaseTool instances, exposed through function
    schemas, and executed only after the model emits a tool-call payload.
    """

    def __init__(
        self,
        llm_name: str,
        enabled_tools: Optional[Iterable[str]] = None,
        direct_tool_call: bool = False,
        web_tool_provider: str = "tavily",
        top_k_results: int = 3,
        max_search_limit: int = 3,
        max_tool_chars: int = 6000,
        max_tool_turns: int = 3,
    ) -> None:
        self.llm_name = llm_name
        self.enabled_tools = normalize_enabled_tools(enabled_tools)
        self.direct_tool_call = direct_tool_call
        self.web_tool_provider = web_tool_provider
        self.top_k_results = max(1, int(top_k_results))
        self.max_search_limit = max(1, int(max_search_limit))
        self.max_tool_chars = max(1000, int(max_tool_chars))
        self.max_tool_turns = max(0, int(max_tool_turns))
        self.registry = self._build_registry()

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool_name in self.enabled_tools:
            if tool_name == "web_search":
                registry.register(
                    WebSearchTool(
                        llm_name=self.llm_name,
                        provider=self.web_tool_provider,
                        top_k_results=self.top_k_results,
                        direct_tool_call=self.direct_tool_call,
                        max_tool_chars=self.max_tool_chars,
                    )
                )
            elif tool_name == "code_generator":
                registry.register(
                    CodeGeneratorTool(
                        llm_name=self.llm_name,
                        direct_tool_call=self.direct_tool_call,
                        max_tool_chars=self.max_tool_chars,
                    )
                )
            elif tool_name == "text_inspector":
                registry.register(
                    TextInspectorTool(
                        llm_name=self.llm_name,
                        direct_tool_call=self.direct_tool_call,
                        max_tool_chars=self.max_tool_chars,
                    )
                )
        return registry

    @property
    def tool_names(self) -> List[str]:
        return self.registry.list_tools()

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_names)

    def run(self, raw_inputs: Dict[str, Any]) -> List[ToolObservation]:
        """Execute all registered tools once for callers that request eager observations."""
        observations: List[ToolObservation] = []
        for tool_name in self.registry.list_tools():
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            result = tool.execute(raw_inputs=raw_inputs)
            observations.append(ToolObservation.from_result(tool_name, result))
        return observations

    def format_observations(self, observations: List[ToolObservation]) -> str:
        if not observations:
            return ""
        sections = []
        for observation in observations:
            content = _truncate_text(observation.content, self.max_tool_chars)
            if content:
                sections.append(f"[{observation.name}]\n{content}")
        if not sections:
            return ""
        return "\n\n".join(sections)

    def format_tool_instructions(self) -> str:
        if not self.registry.list_tools():
            return ""
        schemas = json.dumps(self.registry.get_all_schemas(), indent=2, ensure_ascii=False)
        return (
            "Available tools are defined by the following function schemas:\n"
            f"<tools>\n{schemas}\n</tools>"
        )

    def execute_tool_call(self, tool_call: Dict[str, Any], raw_inputs: Dict[str, Any]) -> List[ToolObservation]:
        normalized = normalize_tool_call(tool_call)
        if normalized is None:
            return [ToolObservation("tool", f"Invalid tool call payload: {tool_call}")]

        name = normalized["name"]
        arguments = normalized["arguments"]
        tool = self.registry.get(name)
        if tool is None:
            enabled = ", ".join(self.registry.list_tools()) if self.registry.list_tools() else "none"
            return [ToolObservation(name, f"Tool '{name}' is not enabled. Enabled tools: {enabled}.")]

        if name == "web_search":
            current_count = int(raw_inputs.get("metadata", {}).get("tool_counts", {}).get("web_search", 0))
            if current_count >= self.max_search_limit:
                return [
                    ToolObservation("web_search", f"web_search call limit reached ({self.max_search_limit}).")
                ]

        result = tool.execute(raw_inputs=raw_inputs, **arguments)
        return [ToolObservation.from_result(name, result)]


def tool_config_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(DEFAULT_TOOL_CONFIG)
    explicit = kwargs.get("tool_config")
    if isinstance(explicit, dict):
        config.update(explicit)
    for key in DEFAULT_TOOL_CONFIG:
        if key in kwargs and kwargs[key] is not None:
            config[key] = kwargs[key]
    config["enabled_tools"] = normalize_enabled_tools(config.get("enabled_tools"))
    return config


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", str(text), re.DOTALL)
    if matches:
        tool_call = _loads_json_tool_call(matches[-1].strip())
        if tool_call is not None:
            return tool_call

    function_calls = re.search(r"<function_calls>(.*?)</function_calls>", str(text), re.DOTALL)
    if function_calls:
        for line in function_calls.group(1).splitlines():
            parsed = _parse_pythonic_call(line.strip())
            if parsed is not None:
                return parsed

    stripped = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    if "</think>" in stripped:
        stripped = re.sub(r"^.*?</think>", "", stripped, flags=re.DOTALL).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{"tool_call"\s*:', stripped):
        try:
            payload, _ = decoder.raw_decode(stripped, match.start())
        except json.JSONDecodeError:
            continue
        tool_call = payload.get("tool_call") if isinstance(payload, dict) else None
        normalized = normalize_tool_call(tool_call)
        if normalized is not None:
            return normalized

    for fence_match in re.finditer(r"```(?:\w+)?\s*(\{.*?\})\s*```", stripped, re.DOTALL):
        tool_call = _loads_json_tool_call(fence_match.group(1).strip())
        if tool_call is not None:
            return tool_call

    for raw_match in reversed(
        list(re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{.*?\}[^{}]*\}', stripped, re.DOTALL))
    ):
        tool_call = _loads_json_tool_call(raw_match.group(0).strip())
        if tool_call is not None:
            return tool_call

    return None


def normalize_tool_call(tool_call: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool_call, dict):
        return None
    if isinstance(tool_call.get("tool_call"), dict):
        tool_call = tool_call["tool_call"]
    if isinstance(tool_call.get("function"), dict):
        function = tool_call["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = tool_call.get("name") or tool_call.get("tool") or tool_call.get("tool_name")
        arguments = tool_call.get("arguments", tool_call.get("args", {}))
    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}
    elif arguments is None:
        arguments = {}
    elif not isinstance(arguments, dict):
        arguments = {"input": arguments}
    return {"name": str(name).strip(), "arguments": arguments}


def _loads_json_tool_call(payload: str) -> Optional[Dict[str, Any]]:
    try:
        tool_call = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return normalize_tool_call(tool_call)


def _parse_pythonic_call(line: str) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    line = re.sub(r"\btrue\b", "True", line)
    line = re.sub(r"\bfalse\b", "False", line)
    line = re.sub(r"\bnull\b", "None", line)
    try:
        tree = ast.parse(line, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        return None
    try:
        positional = [ast.literal_eval(arg) for arg in tree.body.args]
        arguments = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords if kw.arg}
    except (TypeError, ValueError):
        return None
    if positional:
        arguments.setdefault("input", positional[0] if len(positional) == 1 else positional)
    return {"name": tree.body.func.id, "arguments": arguments}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        return fenced_match.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _run_async_from_sync(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except BaseException as exc:
            error["exception"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "exception" in error:
        raise error["exception"]
    return result.get("value")


def _execute_python(code: str, timeout_seconds: int = 10) -> str:
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_alarm = None

    def _timeout_handler(signum, frame):
        raise TimeoutError("Python execution timed out")

    if hasattr(os, "alarm"):
        import signal

        old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        globals_dict = {"__builtins__": __builtins__}
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, globals_dict, globals_dict)
    finally:
        if hasattr(os, "alarm"):
            import signal

            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
    output = stdout.getvalue().strip()
    err = stderr.getvalue().strip()
    if err:
        output = f"{output}\nSTDERR:\n{err}".strip()
    return output or "[no stdout]"
