from typing import Any, Dict, List

from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.graph.node import Node
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.tools.benchmark_tools import BenchmarkToolRunner, parse_tool_call, tool_config_from_kwargs


@AgentRegistry.register("BenchmarkAgent")
class BenchmarkAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str | None = None,
        domain: str = "",
        llm_name: str = "",
        tool_config: Dict[str, Any] | None = None,
    ):
        super().__init__(id, "BenchmarkAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.tool_config = tool_config_from_kwargs({"tool_config": tool_config or {}})
        self.tool_runner = BenchmarkToolRunner(llm_name=llm_name, **self.tool_config)

    def _process_inputs(
        self,
        raw_inputs: Dict[str, Any],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ) -> List[Any]:
        question = raw_inputs.get("question") or raw_inputs.get("task") or ""

        system_parts = [self.constraint]
        tool_instructions = self._tool_instructions()
        if tool_instructions:
            system_parts.append(tool_instructions)

        user_parts = [f"Question:\n{question}"]

        if spatial_info:
            user_parts.append("Other agents' current-round outputs:\n" + _format_agent_info(spatial_info))
        if temporal_info:
            user_parts.append("Previous-round memory:\n" + _format_agent_info(temporal_info))

        if self.tool_runner.enabled_tools:
            user_parts.append(
                "Decide whether a tool is needed. If you use a tool, return only one tool call. "
                "If you already know the answer, give the final answer in the required format."
            )
        else:
            user_parts.append("Give the final answer in the required format.")
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    def _execute(self, input: Dict[str, Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self._run_tool_loop(input, messages)

    async def _async_execute(
        self,
        input: Dict[str, Any],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return await self._run_tool_loop_async(input, messages)

    def _tool_instructions(self) -> str:
        tool_info = self.tool_runner.format_tool_instructions()
        if not tool_info:
            return ""
        return (
            f"{tool_info}\n\n"
            "Tool calling protocol:\n"
            "When a tool is needed, respond with exactly one tool call and no final answer:\n"
            '<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>\n'
            "For code execution, prefer passing explicit Python in the code argument when possible:\n"
            '<tool_call>{"name":"code_generator","arguments":{"code":"print(2 + 2)"}}</tool_call>\n'
            "After a tool observation is returned, continue solving. "
            "When ready, stop calling tools and end with the required final answer line."
        )

    def _run_tool_loop(self, raw_inputs: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
        max_turns = self.tool_runner.max_tool_turns if self.tool_runner.enabled_tools else 0
        if max_turns <= 0:
            return _stringify_response(self.llm.gen(messages))

        for _ in range(max_turns):
            output = _stringify_response(self.llm.gen(messages))
            tool_call = parse_tool_call(output)
            if tool_call is None or not self.tool_runner.enabled_tools:
                return output
            messages.append({"role": "assistant", "content": output})
            observations = self.tool_runner.execute_tool_call(tool_call, raw_inputs)
            observations = self.tool_runner.maybe_distill(_question(raw_inputs), observations)
            self._record_tool_usage(raw_inputs, tool_call, observations)
            messages.append({"role": "user", "content": self._format_tool_result(tool_call, observations)})

        messages.append(
            {
                "role": "user",
                "content": "The tool-call budget is exhausted. Give the final answer in the required format.",
            }
        )
        return _stringify_response(self.llm.gen(messages))

    async def _run_tool_loop_async(self, raw_inputs: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
        max_turns = self.tool_runner.max_tool_turns if self.tool_runner.enabled_tools else 0
        if max_turns <= 0:
            return _stringify_response(await self.llm.agen(messages))

        for _ in range(max_turns):
            output = _stringify_response(await self.llm.agen(messages))
            tool_call = parse_tool_call(output)
            if tool_call is None or not self.tool_runner.enabled_tools:
                return output
            messages.append({"role": "assistant", "content": output})
            observations = self.tool_runner.execute_tool_call(tool_call, raw_inputs)
            observations = self.tool_runner.maybe_distill(_question(raw_inputs), observations)
            self._record_tool_usage(raw_inputs, tool_call, observations)
            messages.append({"role": "user", "content": self._format_tool_result(tool_call, observations)})

        messages.append(
            {
                "role": "user",
                "content": "The tool-call budget is exhausted. Give the final answer in the required format.",
            }
        )
        return _stringify_response(await self.llm.agen(messages))

    def _format_tool_result(self, tool_call: Dict[str, Any], observations: List[Any]) -> str:
        tool_name = tool_call.get("name", "tool")
        content = self.tool_runner.format_observations(observations)
        if not content:
            content = f"[{tool_name}]\n(no observation returned)"
        return (
            f"Tool observation for {tool_name}:\n{content}\n\n"
            "Use this observation to continue. You may call one more tool if needed, "
            "or give the final answer in the required format."
        )

    def _record_tool_usage(self, raw_inputs: Dict[str, Any], tool_call: Dict[str, Any], observations: List[Any]) -> None:
        metadata = raw_inputs.setdefault("metadata", {})
        counts = metadata.setdefault("tool_counts", {})
        history = metadata.setdefault("tool_history", [])
        tool_name = str(tool_call.get("name", "tool"))
        counts[tool_name] = int(counts.get(tool_name, 0)) + 1
        history.append(
            {
                "name": tool_name,
                "arguments": tool_call.get("arguments", {}),
                "result": self.tool_runner.format_observations(observations),
            }
        )


def _format_agent_info(agent_info: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    for agent_id, info in agent_info.items():
        role = info.get("role", "")
        output = info.get("output", "")
        if isinstance(output, list):
            output = "\n".join(str(item) for item in output)
        if not output or output == "None.":
            continue
        parts.append(f"Agent {agent_id} ({role}):\n{output}")
    return "\n\n".join(parts)


def _question(raw_inputs: Dict[str, Any]) -> str:
    return str(raw_inputs.get("question") or raw_inputs.get("task") or "")


def _stringify_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "\n".join(str(item) for item in response)
    return str(response)
