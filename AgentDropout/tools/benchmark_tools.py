import ast
import asyncio
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

CODE_EXECUTION_TIMEOUT_SECONDS = 10 * 60


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


def _clean_path_text(path_like: Any) -> str:
    if path_like is None:
        return ""
    return str(path_like).strip().strip('"').strip("'")


def _path_suffix(path_like: Any) -> str:
    if not path_like:
        return ""
    return Path(str(path_like)).suffix.lower()


def _normalize_web_provider(provider: str) -> str:
    normalized = str(provider or "tavily").strip().lower()
    if normalized == "serper":
        return "searchapi"
    return normalized


def _web_provider_unavailable_reason(provider: str) -> Optional[str]:
    provider = _normalize_web_provider(provider)
    if provider == "wiki":
        try:
            from AgentDropout.tools.search import wiki as wiki_module
        except Exception as exc:
            return f"Wikipedia search module is unavailable: {exc}"
        return None if getattr(wiki_module, "wikipedia", None) is not None else "wikipedia package is not installed"
    if provider == "tavily":
        return None if os.getenv("TAVILY_API_KEY") else "TAVILY_API_KEY is not set"
    if provider == "searchapi":
        return None if os.getenv("SEARCHAPI_API_KEY") else "SEARCHAPI_API_KEY is not set"
    if provider == "google":
        missing = [name for name in ("GOOGLE_API_KEY", "GOOGLE_CSE_ID") if not os.getenv(name)]
        return None if not missing else f"{', '.join(missing)} is not set"
    return f"unsupported web_tool_provider '{provider}'"


def _select_available_web_provider(provider: str) -> Tuple[Optional[str], Optional[str]]:
    requested = _normalize_web_provider(provider)
    requested_reason = _web_provider_unavailable_reason(requested)
    if requested_reason is None:
        return requested, None

    for fallback in ("tavily", "searchapi", "google", "wiki"):
        if fallback == requested:
            continue
        if _web_provider_unavailable_reason(fallback) is None:
            return fallback, f"{requested} unavailable ({requested_reason}); using {fallback}"
    return None, (
        f"{requested} unavailable ({requested_reason}); no credentialed fallback provider is available. "
        "Set TAVILY_API_KEY, SEARCHAPI_API_KEY, or GOOGLE_API_KEY plus GOOGLE_CSE_ID, "
        "or install the wikipedia package and run with --web_tool_provider wiki for Wikipedia-only search."
    )


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
        if isinstance(response, str):
            return response
        if hasattr(response, "full_text"):
            return str(response.full_text)
        if isinstance(response, dict) and "full_text" in response:
            return str(response["full_text"])
        if isinstance(response, list):
            return "\n".join(str(item) for item in response)
        return str(response)

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

    def _attachment_candidates(self, raw_inputs: Dict[str, Any], requested_path: Any) -> List[Any]:
        if requested_path:
            if isinstance(requested_path, (list, tuple)):
                return list(requested_path)
            return [requested_path]

        attachments = raw_inputs.get("attachments") or raw_inputs.get("metadata", {}).get("attachments") or []
        if isinstance(attachments, (list, tuple)):
            return list(attachments)
        return [attachments]

    def _candidate_path_value(self, candidate: Any) -> Any:
        if isinstance(candidate, dict):
            for key in ("path", "file_path", "full_file_path", "file", "filename", "attachment"):
                value = candidate.get(key)
                if value:
                    return value
            return ""
        return candidate

    def _resolve_attachment_path(self, candidate: Any, raw_inputs: Dict[str, Any]) -> Tuple[Optional[Path], str]:
        value = self._candidate_path_value(candidate)
        candidate_text = _clean_path_text(value)
        if not candidate_text:
            return None, ""

        path = Path(candidate_text).expanduser()
        if path.exists():
            return path, candidate_text

        if candidate_text.startswith("./"):
            absolute_text = candidate_text[1:]
            absolute_path = Path(absolute_text).expanduser()
            if absolute_path.is_absolute() and absolute_path.exists():
                return absolute_path, candidate_text

        metadata = raw_inputs.get("metadata", {}) if isinstance(raw_inputs.get("metadata"), dict) else {}
        for source in (raw_inputs, metadata):
            full_file_path = source.get("full_file_path") if isinstance(source, dict) else None
            if not full_file_path:
                continue
            full_path = Path(str(full_file_path)).expanduser()
            source_names = {
                _clean_path_text(source.get("file_name", "")),
                _clean_path_text(source.get("file_path", "")),
                _clean_path_text(full_file_path),
            }
            if candidate_text in source_names or Path(candidate_text).name in source_names:
                if full_path.exists():
                    return full_path, candidate_text
        return None, candidate_text

    def _infer_attachment_suffix(
        self,
        candidate: Any,
        raw_inputs: Dict[str, Any],
        actual_path: Optional[Path],
    ) -> str:
        suffix = _path_suffix(actual_path)
        if suffix:
            return suffix

        values: List[Any] = []
        if isinstance(candidate, dict):
            values.extend(candidate.get(key) for key in ("file_name", "filename", "file_path", "path"))
        values.append(self._candidate_path_value(candidate))

        metadata = raw_inputs.get("metadata", {}) if isinstance(raw_inputs.get("metadata"), dict) else {}
        for source in (raw_inputs, metadata):
            if isinstance(source, dict):
                values.extend(source.get(key) for key in ("file_name", "filename", "file_path", "path"))

        for value in values:
            suffix = _path_suffix(value)
            if suffix:
                return suffix
        return ""

    def _read_attachment(self, file_path: Path, suffix: str, task: str) -> str:
        from AgentDropout.tools.reader.readers import IMGReader, READER_MAP, VideoReader

        suffix = suffix.lower()
        if suffix not in READER_MAP:
            raise ValueError(f"unsupported file type '{suffix or 'unknown'}'")

        reader = READER_MAP[suffix]
        def parse_with_reader(path_to_read: Path) -> Any:
            path_text = str(path_to_read)
            if isinstance(reader, (IMGReader, VideoReader)):
                return reader.parse(path_text, task)
            return reader.parse(path_text)

        if _path_suffix(file_path) == suffix:
            file_content = parse_with_reader(file_path)
        else:
            with tempfile.TemporaryDirectory(prefix="agentprune_attachment_") as temp_dir:
                alias_path = Path(temp_dir) / f"attachment{suffix}"
                try:
                    alias_path.symlink_to(file_path)
                except OSError:
                    shutil.copyfile(file_path, alias_path)
                file_content = parse_with_reader(alias_path)

        suffix_name = suffix.lstrip(".")
        if isinstance(file_content, tuple):
            parts = [f"\nThe {suffix_name} file contains:\n---\n{file_content[0]}"]
            if len(file_content) > 1 and file_content[1]:
                parts.append(f"\nExecution result:\n{file_content[1]}")
            if len(file_content) > 2 and file_content[2]:
                parts.append(f"\nExecution error message:\n{file_content[2]}")
            parts.append("\n---")
            return "".join(parts)
        return f"\nThe {suffix_name} file contains:\n---\n{file_content}\n---"

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
        attachments = self._attachment_candidates(raw_inputs, requested_path)
        try:
            from AgentDropout.tools.reader.readers import READER_MAP  # noqa: F401
        except ModuleNotFoundError as exc:
            if attachments:
                return ToolResult(success=False, output="", metadata={}, error=f"text_inspector unavailable: missing dependency {exc.name}")
            return ToolResult(success=True, output="", metadata={"attachments": 0})

        sections = []
        failures = []
        inspected = 0
        task = str(raw_inputs.get("question") or raw_inputs.get("task") or "")
        for candidate in attachments:
            try:
                actual_path, display_path = self._resolve_attachment_path(candidate, raw_inputs)
                if actual_path is None:
                    if display_path:
                        failures.append(f"Attachment not found: {display_path}")
                    continue
                suffix = self._infer_attachment_suffix(candidate, raw_inputs, actual_path)
                content = self._read_attachment(actual_path, suffix, task)
                if content:
                    inspected += 1
                    sections.append(content)
            except Exception as exc:
                sections.append(f"Failed to inspect {candidate}: {exc}")
        output = "\n\n".join(sections + failures)
        metadata = {"attachments": len(attachments), "inspected": inspected}
        if sections:
            return ToolResult(success=True, output=self._truncate(output), metadata=metadata)
        if failures:
            return ToolResult(success=False, output="", metadata=metadata, error=self._truncate(output))
        return ToolResult(success=True, output="", metadata=metadata)


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
        provider = _normalize_web_provider(self.provider)
        unavailable_reason = _web_provider_unavailable_reason(provider)
        if unavailable_reason is not None:
            raise RuntimeError(
                f"{provider} provider unavailable: {unavailable_reason}. "
                "Set the required environment variables or choose another --web_tool_provider."
            )
        if provider == "wiki":
            queries = [query]
            result = _run_async_from_sync(lambda: search_wiki_main(queries))
            return "\n".join(item for item in result if item)
        if provider == "tavily":
            return self._search_tavily(query)
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
        timeout_seconds: int = CODE_EXECUTION_TIMEOUT_SECONDS,
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
        self.unavailable_tools: Dict[str, str] = {}
        self.registry = self._build_registry()

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool_name in self.enabled_tools:
            if tool_name == "web_search":
                provider, unavailable_reason = _select_available_web_provider(self.web_tool_provider)
                if provider is None:
                    self.unavailable_tools["web_search"] = unavailable_reason or "web_search is unavailable"
                    continue
                self.web_tool_provider = provider
                registry.register(
                    WebSearchTool(
                        llm_name=self.llm_name,
                        provider=provider,
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
            if name in self.unavailable_tools:
                return [
                    ToolObservation(
                        name,
                        f"Tool '{name}' is unavailable: {self.unavailable_tools[name]} Enabled tools: {enabled}.",
                    )
                ]
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


_PYTHON_EXEC_WRAPPER = r"""
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

code = sys.stdin.read()
stdout = io.StringIO()
stderr = io.StringIO()
try:
    globals_dict = {"__builtins__": __builtins__}
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exec(code, globals_dict, globals_dict)
except BaseException:
    captured_stderr = stderr.getvalue()
    if captured_stderr:
        sys.stderr.write(captured_stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

sys.stdout.write(stdout.getvalue())
sys.stderr.write(stderr.getvalue())
"""


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


def _execute_python(code: str, timeout_seconds: int = CODE_EXECUTION_TIMEOUT_SECONDS) -> str:
    timeout_seconds = max(1, int(timeout_seconds))
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PYTHON_EXEC_WRAPPER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to start Python subprocess: {exc}") from exc

    try:
        stdout, stderr = process.communicate(input=code, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.communicate()
        raise TimeoutError(f"Python execution timed out after {timeout_seconds} seconds") from exc

    output = (stdout or "").strip()
    err = (stderr or "").strip()
    if process.returncode != 0:
        detail = err or output or "[no output]"
        raise RuntimeError(f"Python execution failed with exit code {process.returncode}: {detail}")
    if err:
        output = f"{output}\nSTDERR:\n{err}".strip()
    return output or "[no stdout]"
