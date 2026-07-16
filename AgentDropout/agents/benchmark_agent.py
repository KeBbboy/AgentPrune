import asyncio
from typing import Any, Dict, List

from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.graph.node import Node
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.llm.price import cal_token
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.tools.benchmark_tools import BenchmarkToolRunner, parse_tool_call, tool_config_from_kwargs
from AgentDropout.utils.globals import get_token_usage_context


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

        if self.tool_runner.has_tools:
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
            '<tool_call>{"name":"tool_name","arguments":{"argument_name":"argument_value"}}</tool_call>\n'
            "Choose a tool name and arguments that match one of the schemas exactly. "
            "After a tool observation is returned, continue solving. "
            "When ready, stop calling tools and end with the required final answer line."
        )

    def _run_tool_loop(self, raw_inputs: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
        max_turns = self.tool_runner.max_tool_turns if self.tool_runner.has_tools else 0
        if max_turns <= 0:
            return _stringify_response(self.llm.gen(messages))

        for _ in range(max_turns):
            output = _stringify_response(self.llm.gen(messages))
            tool_call = parse_tool_call(output)
            if tool_call is None or not self.tool_runner.has_tools:
                return output
            messages.append({"role": "assistant", "content": output})
            observations = self.tool_runner.execute_tool_call(tool_call, raw_inputs)
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
        max_turns = self.tool_runner.max_tool_turns if self.tool_runner.has_tools else 0
        if max_turns <= 0:
            return _stringify_response(await self.llm.agen(messages))

        for _ in range(max_turns):
            output = _stringify_response(await self.llm.agen(messages))
            tool_call = parse_tool_call(output)
            if tool_call is None or not self.tool_runner.has_tools:
                return output
            messages.append({"role": "assistant", "content": output})
            observations = await asyncio.to_thread(self.tool_runner.execute_tool_call, tool_call, raw_inputs)
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
        result_text = self.tool_runner.format_observations(observations)
        result_tokens = int(cal_token(self.llm_name, result_text)) if result_text else 0
        usage_context = get_token_usage_context()
        round_id = str(usage_context.get("round_id", "__unknown__"))
        counts[tool_name] = int(counts.get(tool_name, 0)) + 1
        token_counts = metadata.setdefault("tool_tokens", {})
        token_counts[tool_name] = int(token_counts.get(tool_name, 0)) + result_tokens
        agent_counts = metadata.setdefault("tool_counts_by_agent", {})
        agent_entry = agent_counts.setdefault(
            self.id,
            {
                "id": self.id,
                "role": self.role,
                "agent_name": self.agent_name,
                "tools": {},
                "tool_tokens": {},
                "total": 0,
                "total_tool_tokens": 0,
            },
        )
        agent_entry["role"] = self.role
        agent_entry["agent_name"] = self.agent_name
        agent_entry["tools"][tool_name] = int(agent_entry["tools"].get(tool_name, 0)) + 1
        agent_entry["total"] = int(agent_entry.get("total", 0)) + 1
        agent_entry.setdefault("tool_tokens", {})
        agent_entry["tool_tokens"][tool_name] = int(agent_entry["tool_tokens"].get(tool_name, 0)) + result_tokens
        agent_entry["total_tool_tokens"] = int(agent_entry.get("total_tool_tokens", 0)) + result_tokens
        round_entry = agent_entry.setdefault("by_round", {}).setdefault(
            round_id,
            {
                "round_id": round_id,
                "tools": {},
                "tool_tokens": {},
                "total": 0,
                "total_tool_tokens": 0,
            },
        )
        round_entry["tools"][tool_name] = int(round_entry["tools"].get(tool_name, 0)) + 1
        round_entry["total"] = int(round_entry.get("total", 0)) + 1
        round_entry["tool_tokens"][tool_name] = int(round_entry["tool_tokens"].get(tool_name, 0)) + result_tokens
        round_entry["total_tool_tokens"] = int(round_entry.get("total_tool_tokens", 0)) + result_tokens
        history.append(
            {
                "agent_id": self.id,
                "agent_role": self.role,
                "agent_name": self.agent_name,
                "round_id": round_id,
                "node_type": usage_context.get("node_type"),
                "sample_run_id": usage_context.get("sample_run_id"),
                "name": tool_name,
                "result_tokens": result_tokens,
                "arguments": tool_call.get("arguments", {}),
                "result": result_text,
            }
        )


def _format_agent_info(agent_info: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    
    def _get_output_content(output):
        if isinstance(output, dict) and 'full_text' in output:
            return output['full_text']
        return output
    
    for agent_id, info in agent_info.items():
        role = info.get("role", "")
        output = _get_output_content(info.get("output", ""))
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
    if isinstance(response, dict) and 'full_text' in response:
        return response['full_text']
    if hasattr(response, 'full_text'):
        return response.full_text
    if isinstance(response, list):
        return "\n".join(str(item) for item in response)
    return str(response)
