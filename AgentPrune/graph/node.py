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
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.outputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'outputs':[]}
        self.conversation_history : List[Dict] = [] # chat history of the whole conversation
        self.last_input_messages: Optional[List[Dict[str, Any]]] = None
        self.last_llm_usage: Optional[Dict[str, int]] = None
        self.active_spatial_message_modes: Dict[str, str] = {}
        self.active_temporal_message_modes: Dict[str, str] = {}
        self.runtime_sample_index: Optional[int] = None
        self.runtime_round_index: Optional[int] = None
        self.runtime_execution_index: Optional[int] = None

    @property
    def node_name(self):
        return self.__class__.__name__

    def set_last_input_messages(self, messages: Optional[List[Dict[str, Any]]]):
        self.last_input_messages = messages

    def reset_runtime_audit(self):
        self.runtime_sample_index = None
        self.runtime_round_index = None
        self.runtime_execution_index = None
    
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
    
    def update_memory(self):
        self.last_memory['outputs'] = self.outputs

    def _extract_answer_tag_content(self, predecessor_output: str) -> str:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            predecessor_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if answer_match is None:
            return ""
        return answer_match.group(1).strip()

    def _extract_first_line_answer(self, predecessor_output: Any) -> str:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)

        answer_content = self._extract_answer_tag_content(predecessor_output)
        if answer_content:
            return answer_content

        for line in predecessor_output.splitlines():
            line = line.strip()
            if line:
                return line
        return predecessor_output.strip()

    def _extract_reasoning_without_answer(self, predecessor_output: Any) -> str:
        if not isinstance(predecessor_output, str):
            predecessor_output = str(predecessor_output)

        reasoning_output = re.sub(
            r"<answer>\s*.*?\s*</answer>",
            "",
            predecessor_output,
            flags=re.IGNORECASE | re.DOTALL,
        )

        kept_lines = []
        first_content_line_seen = False
        for line in reasoning_output.splitlines():
            stripped_line = line.strip()
            if not first_content_line_seen and stripped_line:
                first_content_line_seen = True
                if re.match(r"^[A-E]$", stripped_line, flags=re.IGNORECASE):
                    continue
            if re.match(r"^(final\s+)?answer\s*(is|:)\s*.+$", stripped_line, flags=re.IGNORECASE):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _format_predecessor_output(self, predecessor_output: Any, message_type: str) -> str:
        if message_type == "answer_only":
            return self._extract_first_line_answer(predecessor_output)
        if message_type == "reasoning_only":
            return self._extract_reasoning_without_answer(predecessor_output)
        return predecessor_output

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
                message_type = self.active_spatial_message_modes.get(predecessor.id, "full")
                predecessor_output = self._format_predecessor_output(predecessor_output, message_type)
                spatial_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": predecessor_output,
                    "message_type": message_type,
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
                message_type = self.active_temporal_message_modes.get(predecessor.id, "full")
                predecessor_output = self._format_predecessor_output(predecessor_output, message_type)
                temporal_info[predecessor.id] = {
                    "role": predecessor.role,
                    "output": predecessor_output,
                    "message_type": message_type,
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
