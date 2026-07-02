from typing import List,Any,Dict

from AgentDropout.graph.node import Node
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.tools.coding.python_executor import execute_code_get_return

@AgentRegistry.register('MathSolver_aqua')
class MathSolver_aqua(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolver_aqua" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        
        def _get_output_content(output):
            if isinstance(output, dict) and 'full_text' in output:
                return output['full_text']
            return output
        
        for id, info in spatial_info.items():
            output_content = _get_output_content(info['output'])
            spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{output_content}\n\n"
        for id, info in temporal_info.items():
            output_content = _get_output_content(info['output'])
            temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{output_content}\n\n"
        user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        response_text = str(response)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response_text.lstrip("```python\n").rstrip("\n```"))
            response_text += f"\nthe answer is {answer}"
        return response_text
