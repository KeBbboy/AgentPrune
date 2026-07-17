import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio
import re

from AgentDropout.graph.message_compression import (
    aggregate_node_aware,
    compress_messages_role_aware,
)
from AgentDropout.utils.globals import (
    get_token_usage_context,
    reset_token_usage_context,
    set_token_usage_context,
)


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        

    def bind_to_graph(self, graph) -> None:
        """绑定所属 Graph, 让消息压缩钩子读取图级配置。"""
        self.graph = graph
    
    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def _split_answer_and_reasoning(self, predecessor_output: str) -> tuple[str, str]:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)
        if self.domain == "humaneval":
            return self._split_humaneval_answer_and_reasoning(predecessor_output)

        lines = predecessor_output.splitlines()
        non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
        if len(non_empty_indices) == 0:
            return "", ""

        answer_idx = non_empty_indices[0]
        answer = lines[answer_idx].strip()
        reasoning = "\n".join(lines[answer_idx + 1:]).strip()
        return answer, reasoning

    def _split_humaneval_answer_and_reasoning(self, predecessor_output: str) -> tuple[str, str]:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            predecessor_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if answer_match is not None:
            answer = answer_match.group(1).strip()
            reasoning = (
                predecessor_output[:answer_match.start()] + "\n" + predecessor_output[answer_match.end():]
            ).strip()
            return answer, reasoning
        return predecessor_output.strip(), ""

    def _extract_answer_content(self, predecessor_output: str) -> str:
        answer, _ = self._split_answer_and_reasoning(predecessor_output)
        return answer

    def _extract_reasoning_content(self, predecessor_output: str) -> str:
        _, reasoning = self._split_answer_and_reasoning(predecessor_output)
        return reasoning

    def _message_compression_enabled(self) -> bool:
        cfg = getattr(getattr(self, "graph", None), "ours_message_config", None)
        return bool(
            cfg
            and (
                getattr(cfg, "role_aware_compression", False)
                or getattr(cfg, "node_aware_aggregation", False)
            )
        )

    def _rehydrate_full_to_structured(self, output: Any) -> Dict[str, Any] | None:
        """把压扁后的 full_text 还原为 structured payload, 供消息压缩使用。"""
        if not isinstance(output, str) or not output:
            return None

        think_match = re.search(r"<(?:redacted_)?think>(.*?)</(?:redacted_)?think>", output, re.DOTALL)
        if think_match is not None:
            reasoning = think_match.group(1).strip()
            answer = (output[: think_match.start()] + output[think_match.end():]).strip()
        else:
            answer, reasoning = self._split_answer_and_reasoning(output)

        if not answer and not reasoning:
            return None
        return {
            "reasoning": reasoning,
            "answer": answer,
            "full_text": output,
        }

    def _resolve_message_payload(self, predecessor: 'Node', output: Any, source: str = "spatial") -> tuple[str, Any]:
        if hasattr(output, 'reasoning') and hasattr(output, 'answer') and hasattr(output, 'full_text'):
            return "structured", {
                "reasoning": output.reasoning,
                "answer": output.answer,
                "full_text": output.full_text,
            }
        if self._message_compression_enabled():
            structured = self._rehydrate_full_to_structured(output)
            if structured is not None:
                return "structured", structured
        return "full", output

    def get_spatial_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                message_type, selected_output = self._resolve_message_payload(predecessor, predecessor_output, source="spatial")
                spatial_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": selected_output,
                    "message_type": message_type,
                }

        return self._apply_message_compression_hooks(spatial_info)

    def get_temporal_info(self, raw_inputs: Any = None)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                message_type, selected_output = self._resolve_message_payload(predecessor, predecessor_output, source="temporal")
                temporal_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": selected_output,
                    "message_type": message_type,
                }

        return self._apply_message_compression_hooks(temporal_info)

    async def aget_temporal_info(self, raw_inputs: Any = None)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                message_type, selected_output = self._resolve_message_payload(predecessor, predecessor_output, source="temporal")
                temporal_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": selected_output,
                    "message_type": message_type,
                }

        return self._apply_message_compression_hooks(temporal_info)

    def _apply_message_compression_hooks(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """按图级配置应用 role-aware 压缩和 node-aware 聚合。"""
        cfg = getattr(getattr(self, "graph", None), "ours_message_config", None)
        if not cfg or not info:
            return info

        if getattr(cfg, "node_aware_aggregation", False):
            info = aggregate_node_aware(info, cfg, getattr(self, "role", ""))
        if getattr(cfg, "role_aware_compression", False):
            info = compress_messages_role_aware(info, cfg, getattr(self, "role", ""))

        for payload in info.values():
            if not isinstance(payload, dict) or "output" in payload:
                continue
            if isinstance(payload.get("full_text"), str):
                payload["output"] = payload["full_text"]
                continue
            parts = []
            if isinstance(payload.get("reasoning"), str):
                parts.append(payload["reasoning"])
            if isinstance(payload.get("answer"), str):
                parts.append(payload["answer"])
            payload["output"] = "\n".join(parts)
        return info
    
    def execute(self, input:Any, **kwargs):
        self.outputs = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info(raw_inputs=input)
        token_context = get_token_usage_context()
        token_context.update(
            {
                "agent_id": self.id,
                "agent_role": self.role,
                "agent_name": self.agent_name,
                "node_name": self.node_name,
                "node_type": kwargs.get("node_type", "agent"),
                "round_id": kwargs.get("round_id"),
            }
        )
        context_token = set_token_usage_context(token_context)
        try:
            results = [self._execute(input, spatial_info, temporal_info, **kwargs)]
        finally:
            reset_token_usage_context(context_token)

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs


    async def async_execute(self, input:Any, **kwargs):

        self.outputs = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = await self.aget_temporal_info(raw_inputs=input)
        # print(temporal_info)
        token_context = get_token_usage_context()
        token_context.update(
            {
                "agent_id": self.id,
                "agent_role": self.role,
                "agent_name": self.agent_name,
                "node_name": self.node_name,
                "node_type": kwargs.get("node_type", "agent"),
                "round_id": kwargs.get("round_id"),
            }
        )
        context_token = set_token_usage_context(token_context)
        try:
            tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            reset_token_usage_context(context_token)
        for result in results:
            # Guarantee ``outputs`` is a list of strings (the legacy contract
            # used by ``graph.arun`` and downstream evaluators).  When an
            # ``LLMResponse`` is returned we keep the full ``<think>...reasoning
            # ...answer...`` text (i.e. ``full_text``) so that:
            #   * postprocess helpers can still find ``The answer is ...`` /
            #     ``Answer: X`` / ``<answer>...</answer>`` markers (which only
            #     live inside the final answer block),
            #   * ``has_<think>`` / ``reasoning_len`` / ``full_len`` diagnostics
            #     reflect the real model output rather than collapsing to 0,
            #   * ``gsm_get_predict``'s regex fallback works on the actual
            #     response instead of silently extracting ``str(LLMResponse)``.
            # Plain strings and other objects are kept verbatim.
            if isinstance(result, list):
                flattened: List[Any] = []
                for item in result:
                    if hasattr(item, "full_text") and isinstance(
                        getattr(item, "full_text", None), str
                    ):
                        flattened.append(getattr(item, "full_text"))
                    elif isinstance(item, str):
                        flattened.append(item)
                    else:
                        flattened.append(str(item))
                self.outputs.extend(flattened)
            else:
                if hasattr(result, "full_text") and isinstance(
                    getattr(result, "full_text", None), str
                ):
                    self.outputs.append(getattr(result, "full_text"))
                elif isinstance(result, str):
                    self.outputs.append(result)
                else:
                    self.outputs.append(str(result))
        return self.outputs
               
    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
