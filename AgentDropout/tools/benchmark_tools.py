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


def normalize_enabled_tools(tools: Optional[Iterable[str]]) -> List[str]:
    if tools is None:
        return []
    normalized = []
    for tool in tools:
        tool_name = str(tool).strip()
        if tool_name and tool_name not in normalized:
            normalized.append(tool_name)
    return normalized


class BenchmarkToolRunner:
    """Small tool layer matching the msc-thesis benchmark tool names.

    Tools are made available to the BenchmarkAgent, but they are only executed
    after the model emits a tool-call payload. `direct_tool_call=False`
    optionally asks the LLM to summarize raw tool output before it is returned
    to the answering loop, approximating msc-thesis sub-agent mode.
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
        self._llm = None
        self._search_engine = None

    def run(self, raw_inputs: Dict[str, Any]) -> List[ToolObservation]:
        """Legacy eager execution path kept for compatibility."""
        observations: List[ToolObservation] = []
        if "text_inspector" in self.enabled_tools:
            observations.extend(self._run_text_inspector(raw_inputs))
        if "web_search" in self.enabled_tools:
            observations.extend(self._run_web_search(raw_inputs))
        if "code_generator" in self.enabled_tools:
            observations.extend(self._run_code_generator(raw_inputs))
        return observations

    def format_observations(self, observations: List[ToolObservation]) -> str:
        if not observations:
            return ""
        sections = []
        for observation in observations:
            content = self._truncate(observation.content)
            if content:
                sections.append(f"[{observation.name}]\n{content}")
        if not sections:
            return ""
        return "\n\n".join(sections)

    def format_tool_instructions(self) -> str:
        if not self.enabled_tools:
            return ""
        descriptions = {
            "web_search": (
                "Search the web for current or external facts. "
                'Arguments: {"query": "concise search query"}.'
            ),
            "code_generator": (
                "Run a short Python calculation or verification. "
                'Arguments: {"code": "python code"} or {"prompt": "calculation request"}.'
            ),
            "text_inspector": (
                "Inspect an attached local file. "
                'Arguments: {"path": "optional file path"}; omit path to inspect provided attachments.'
            ),
        }
        lines = ["Available tools:"]
        for tool_name in self.enabled_tools:
            description = descriptions.get(tool_name, "No description available.")
            lines.append(f"- {tool_name}: {description}")
        return "\n".join(lines)

    def execute_tool_call(self, tool_call: Dict[str, Any], raw_inputs: Dict[str, Any]) -> List[ToolObservation]:
        normalized = normalize_tool_call(tool_call)
        if normalized is None:
            return [ToolObservation("tool", f"Invalid tool call payload: {tool_call}")]

        name = normalized["name"]
        arguments = normalized["arguments"]
        if name not in self.enabled_tools:
            enabled = ", ".join(self.enabled_tools) if self.enabled_tools else "none"
            return [ToolObservation(name, f"Tool '{name}' is not enabled. Enabled tools: {enabled}.")]

        if name == "web_search":
            current_count = int(raw_inputs.get("metadata", {}).get("tool_counts", {}).get("web_search", 0))
            if current_count >= self.max_search_limit:
                return [
                    ToolObservation("web_search", f"web_search call limit reached ({self.max_search_limit}).")
                ]
            query = _first_argument(arguments, "query", "q", "search_query", "sub_goal", "input", "question")
            return self._run_web_search(raw_inputs, query=query)

        if name == "code_generator":
            code = _first_argument(arguments, "code", "python_code", "program")
            prompt = _first_argument(arguments, "prompt", "query", "sub_goal", "task", "question", "input")
            return self._run_code_generator(raw_inputs, prompt_override=prompt, code=code)

        if name == "text_inspector":
            requested_path = _first_argument(arguments, "path", "file_path", "file", "filename", "attachment")
            return self._run_text_inspector(raw_inputs, requested_path=requested_path)

        return [ToolObservation(name, f"Unsupported tool '{name}'.")]

    def _run_text_inspector(self, raw_inputs: Dict[str, Any], requested_path: Any = None) -> List[ToolObservation]:
        observations = []
        attachments = raw_inputs.get("attachments") or raw_inputs.get("metadata", {}).get("attachments") or []
        if requested_path:
            if isinstance(requested_path, (list, tuple)):
                attachments = list(requested_path)
            else:
                attachments = [requested_path]
        try:
            from AgentDropout.tools.reader.readers import GeneralReader
        except ModuleNotFoundError as exc:
            if attachments:
                return [ToolObservation("text_inspector", f"text_inspector unavailable: missing dependency {exc.name}")]
            return []
        reader = GeneralReader()
        for path in attachments:
            try:
                if not path or not Path(path).exists():
                    continue
                content = reader.read(str(raw_inputs.get("question") or raw_inputs.get("task") or ""), str(path))
                observations.append(ToolObservation("text_inspector", content))
            except Exception as exc:
                observations.append(ToolObservation("text_inspector", f"Failed to inspect {path}: {exc}"))
        return observations

    def _run_web_search(self, raw_inputs: Dict[str, Any], query: Optional[str] = None) -> List[ToolObservation]:
        question = str(raw_inputs.get("question") or raw_inputs.get("task") or "").strip()
        if not question and not query:
            return []
        query = str(query or "").strip() or self._build_search_query(question)
        if not query:
            return []
        try:
            content = self._search(query)
        except Exception as exc:
            content = f"Search failed for query '{query}': {exc}"
        return [ToolObservation("web_search", content)] if content else []

    def _run_code_generator(
        self,
        raw_inputs: Dict[str, Any],
        prompt_override: Optional[str] = None,
        code: Optional[str] = None,
    ) -> List[ToolObservation]:
        question = str(raw_inputs.get("question") or raw_inputs.get("task") or "").strip()
        if not question and not prompt_override and not code:
            return []
        try:
            if code is None:
                prompt = (
                    "Write a short Python calculation, transformation, or verification to help solve the problem. "
                    "If code is not useful, return only the word SKIP. "
                    "Do not use internet or files unless the prompt explicitly contains a local file path. "
                    "Print the useful result.\n\n"
                    f"Problem:\n{prompt_override or question}"
                )
                code = self._llm_complete(
                    "You generate minimal Python snippets for benchmark problem solving. "
                    "Return only Python code or SKIP.",
                    prompt,
                )
            code = _strip_code_fences(code).strip()
            if not code or code.upper().startswith("SKIP"):
                return []
            output = _execute_python(code)
            return [ToolObservation("code_generator", f"Code:\n{code}\n\nOutput:\n{output}")]
        except Exception as exc:
            return [ToolObservation("code_generator", f"Code helper failed: {exc}")]

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
        provider = self.web_tool_provider.lower()
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


def _first_argument(arguments: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = arguments.get(key)
        if value not in (None, ""):
            return value
    return None


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
