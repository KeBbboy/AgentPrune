"""msc-thesis style tool abstractions for benchmark tool execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardized result returned by every benchmark tool."""

    success: bool
    output: str
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class BaseTool(ABC):
    """Abstract base class matching the msc-thesis tool interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier used in model tool calls."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description used in tool schemas."""
        raise NotImplementedError

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return a Qwen3-compatible function schema."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with model-provided arguments."""
        raise NotImplementedError

    def validate_args(self, **kwargs) -> bool:
        return True

    def cleanup(self) -> None:
        pass


class ToolRegistry:
    """Central registry for benchmark tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def cleanup_all(self) -> None:
        for tool in self._tools.values():
            tool.cleanup()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
