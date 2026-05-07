import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio
import re


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
        self.pruned_spatial_predecessors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.pruned_temporal_predecessors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}
        self.conversation_history : List[Dict] = [] # chat history of the whole conversation
        self.last_input_messages: Optional[List[Dict[str, Any]]] = None
        self.last_llm_usage: Optional[Dict[str, int]] = None
        self.pruned_edge_patch_mode: str = "none"
        self.pruned_temporal_patch_mode: str = "none"
        self.active_edge_patch_mode: str = "none"
        self.active_temporal_patch_mode: str = "none"
        self.truncate_content_ratio: float = 1.0
        self.pruned_edge_use_original_prompt: bool = False
        self.pruned_temporal_use_original_prompt: bool = False
        self.restored_edge_patch_mode: str = "none"
        self.restored_temporal_patch_mode: str = "none"
        self.restored_spatial_predecessor_ids: set[str] = set()
        self.restored_temporal_predecessor_ids: set[str] = set()
        self.enable_pruned_delivery_audit: bool = False
        self.runtime_sample_index: Optional[int] = None
        self.runtime_round_index: Optional[int] = None
        self.runtime_execution_index: Optional[int] = None
        self.transmission_audit_events: List[Dict[str, Any]] = []

    @property
    def node_name(self):
        return self.__class__.__name__

    def set_last_input_messages(self, messages: Optional[List[Dict[str, Any]]]):
        self.last_input_messages = messages

    def reset_runtime_audit(self):
        self.runtime_sample_index = None
        self.runtime_round_index = None
        self.runtime_execution_index = None
        self.transmission_audit_events = []

    def _record_pruned_delivery_event(
        self,
        *,
        predecessor: 'Node',
        edge_type: str,
        source_output_state: str,
        source_output_available: bool,
        delivered: bool,
        failure_reason: Optional[str] = None,
        patch_mode: str = "none",
        original_prompt_format: bool = False,
        delivered_output: Optional[Any] = None,
    ) -> None:
        if not self.enable_pruned_delivery_audit:
            return
        delivered_text = ""
        if delivered and delivered_output is not None:
            delivered_text = delivered_output if isinstance(delivered_output, str) else str(delivered_output)
        self.transmission_audit_events.append({
            "sample_index": self.runtime_sample_index,
            "round_index": self.runtime_round_index,
            "round_number": (self.runtime_round_index + 1) if self.runtime_round_index is not None else None,
            "round_label": (
                f"round_{self.runtime_round_index + 1}"
                if self.runtime_round_index is not None else None
            ),
            "target_execution_index": self.runtime_execution_index,
            "target_id": self.id,
            "target_role": self.role,
            "source_id": predecessor.id,
            "source_role": predecessor.role,
            "source_execution_index": getattr(predecessor, "runtime_execution_index", None),
            "edge_type": edge_type,
            "source_output_state": source_output_state,
            "source_output_available": source_output_available,
            "delivered": delivered,
            "failure_reason": failure_reason,
            "patch_mode": patch_mode,
            "original_prompt_format": original_prompt_format,
            "delivered_char_len": len(delivered_text),
            "delivered_word_len": len(delivered_text.split()) if delivered_text else 0,
        })
    
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
        self.pruned_spatial_predecessors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.pruned_temporal_predecessors: List[Node] = []
        self.restored_spatial_predecessor_ids = set()
        self.restored_temporal_predecessor_ids = set()
    
    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def _extract_patched_output(self, predecessor_output: Any, patch_mode: str) -> str:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)
        parts = predecessor_output.rsplit("\n\n", 1)
        if patch_mode == "full":
            patched_output = predecessor_output.strip()
        elif patch_mode == "tail":
            patched_output = parts[-1].strip()
        elif patch_mode == "body":
            patched_output = parts[0].strip() if len(parts) == 2 else predecessor_output.strip()
        elif patch_mode == "answer_only":
            patched_output = self._extract_answer_only(predecessor_output)
        elif patch_mode == "answer_claim":
            patched_output = self._extract_answer_and_claim(predecessor_output)
        elif patch_mode == "answer_steps":
            patched_output = self._extract_answer_and_steps(predecessor_output)
        elif patch_mode == "truncate":
            patched_output = self._extract_truncated_output(predecessor_output)
        else:
            patched_output = predecessor_output.strip()
        return patched_output

    def _extract_xml_block(self, predecessor_output: str, tag: str) -> str:
        block_match = re.search(
            rf"<{tag}>\s*.*?\s*</{tag}>",
            predecessor_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block_match is None:
            return ""
        return block_match.group(0).strip()

    def _extract_answer_only(self, predecessor_output: str) -> str:
        stripped_output = predecessor_output.strip()
        if not stripped_output:
            return ""

        xml_answer_match = re.search(r"<answer>\s*([ABCD])\s*</answer>", stripped_output, flags=re.IGNORECASE)
        if xml_answer_match is not None:
            return xml_answer_match.group(1).upper()

        return stripped_output[0]

    def _extract_answer_and_claim(self, predecessor_output: str) -> str:
        stripped_output = predecessor_output.strip()
        if not stripped_output:
            return ""

        claim_block = self._extract_xml_block(stripped_output, "claim")
        answer_block = self._extract_xml_block(stripped_output, "answer")
        blocks = [block for block in (claim_block, answer_block) if block]
        if blocks:
            return "\n".join(blocks)

        return self._extract_answer_only(stripped_output)

    def _extract_answer_and_steps(self, predecessor_output: str) -> str:
        stripped_output = predecessor_output.strip()
        if not stripped_output:
            return ""

        steps_block = self._extract_xml_block(stripped_output, "steps")
        answer_block = self._extract_xml_block(stripped_output, "answer")
        blocks = [block for block in (steps_block, answer_block) if block]
        if blocks:
            return "\n".join(blocks)

        return self._extract_answer_only(stripped_output)

    def _extract_truncated_output(self, predecessor_output: str) -> str:
        stripped_output = predecessor_output.strip()
        if not stripped_output:
            return ""

        answer_only = self._extract_answer_only(stripped_output)
        xml_answer_match = re.search(r"<answer>\s*[ABCD]\s*</answer>", stripped_output, flags=re.IGNORECASE)
        if xml_answer_match is not None:
            remainder = (stripped_output[:xml_answer_match.start()] + stripped_output[xml_answer_match.end():]).strip()
        else:
            remainder = stripped_output[1:]
        if not remainder.strip():
            return answer_only

        keep_char_count = max(1, int(len(remainder) * self.truncate_content_ratio))
        truncated_remainder = remainder[:keep_char_count].rstrip()
        if answer_only:
            return f"{answer_only}{truncated_remainder}"
        return truncated_remainder

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
                if predecessor.id in self.restored_spatial_predecessor_ids:
                    predecessor_output = self._extract_patched_output(
                        predecessor_output,
                        self.restored_edge_patch_mode,
                    )
                    if not predecessor_output:
                        continue
                elif self.active_edge_patch_mode != "none":
                    predecessor_output = self._extract_patched_output(
                        predecessor_output,
                        self.active_edge_patch_mode,
                    )
                    if not predecessor_output:
                        continue
                spatial_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": predecessor_output,
                    "message_type": "full",
                }

        if self.pruned_spatial_predecessors is not None:
            for predecessor in self.pruned_spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                    source_output_state = "nonempty_list"
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs) == 0:
                    self._record_pruned_delivery_event(
                        predecessor=predecessor,
                        edge_type="spatial",
                        source_output_state="empty_list",
                        source_output_available=False,
                        delivered=False,
                        failure_reason="source_current_round_output_unavailable",
                        patch_mode=self.pruned_edge_patch_mode,
                        original_prompt_format=self.pruned_edge_use_original_prompt,
                    )
                    continue
                else:
                    predecessor_output = predecessor_outputs
                    source_output_state = "scalar_or_other"
                patched_output = self._extract_patched_output(
                    predecessor_output,
                    self.pruned_edge_patch_mode,
                )
                if not patched_output:
                    self._record_pruned_delivery_event(
                        predecessor=predecessor,
                        edge_type="spatial",
                        source_output_state=source_output_state,
                        source_output_available=True,
                        delivered=False,
                        failure_reason="patched_output_empty_after_extraction",
                        patch_mode=self.pruned_edge_patch_mode,
                        original_prompt_format=self.pruned_edge_use_original_prompt,
                    )
                    continue
                self._record_pruned_delivery_event(
                    predecessor=predecessor,
                    edge_type="spatial",
                    source_output_state=source_output_state,
                    source_output_available=True,
                    delivered=True,
                    patch_mode=self.pruned_edge_patch_mode,
                    original_prompt_format=self.pruned_edge_use_original_prompt,
                    delivered_output=patched_output,
                )
                spatial_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": patched_output,
                    "message_type": "full" if self.pruned_edge_use_original_prompt else "pruned_patch",
                    "patch_mode": self.pruned_edge_patch_mode,
                }

        return spatial_info

    def get_temporal_info(self)->Dict[str,Any]:
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
                if predecessor.id in self.restored_temporal_predecessor_ids:
                    predecessor_output = self._extract_patched_output(
                        predecessor_output,
                        self.restored_temporal_patch_mode,
                    )
                    if not predecessor_output:
                        continue
                elif self.active_temporal_patch_mode != "none":
                    predecessor_output = self._extract_patched_output(
                        predecessor_output,
                        self.active_temporal_patch_mode,
                    )
                    if not predecessor_output:
                        continue
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        if self.pruned_temporal_predecessors is not None:
            for predecessor in self.pruned_temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                    source_output_state = "nonempty_list"
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    self._record_pruned_delivery_event(
                        predecessor=predecessor,
                        edge_type="temporal",
                        source_output_state="empty_list",
                        source_output_available=False,
                        delivered=False,
                        failure_reason="source_previous_round_output_unavailable",
                        patch_mode=self.pruned_temporal_patch_mode,
                        original_prompt_format=self.pruned_temporal_use_original_prompt,
                    )
                    continue
                else:
                    predecessor_output = predecessor_outputs
                    source_output_state = "scalar_or_other"
                patched_output = self._extract_patched_output(
                    predecessor_output,
                    self.pruned_temporal_patch_mode,
                )
                if not patched_output:
                    self._record_pruned_delivery_event(
                        predecessor=predecessor,
                        edge_type="temporal",
                        source_output_state=source_output_state,
                        source_output_available=True,
                        delivered=False,
                        failure_reason="patched_output_empty_after_extraction",
                        patch_mode=self.pruned_temporal_patch_mode,
                        original_prompt_format=self.pruned_temporal_use_original_prompt,
                    )
                    continue
                self._record_pruned_delivery_event(
                    predecessor=predecessor,
                    edge_type="temporal",
                    source_output_state=source_output_state,
                    source_output_available=True,
                    delivered=True,
                    patch_mode=self.pruned_temporal_patch_mode,
                    original_prompt_format=self.pruned_temporal_use_original_prompt,
                    delivered_output=patched_output,
                )
                temporal_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": patched_output,
                    "message_type": "full" if self.pruned_temporal_use_original_prompt else "pruned_patch",
                    "patch_mode": self.pruned_temporal_patch_mode,
                }
        
        return temporal_info
    
    def execute(self, input:Any, **kwargs):
        self.outputs = []
        self.last_input_messages = None
        self.last_llm_usage = None
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs


    async def async_execute(self, input:Any, **kwargs):
        self.outputs = []
        self.last_input_messages = None
        self.last_llm_usage = None
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
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
