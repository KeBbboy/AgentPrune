from typing import List,Any,Dict
import re
import asyncio

from AgentPrune.graph.node import Node
from AgentPrune.agents.agent_registry import AgentRegistry
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.search.wiki import search_wiki_main

TRIMMED_SOURCE_ROLES = {"Critic", "Mathematician", "Psychologist", "Historian"}

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None =None, role:str = None, domain: str = "", llm_name: str = "", trim_special_roles_output: bool = False):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""
        self.trim_special_roles_output = trim_special_roles_output

    def _format_shared_output(self, info: Dict[str, str]) -> str:
        output = info["output"]
        if not self.trim_special_roles_output or info["role"] not in TRIMMED_SOURCE_ROLES:
            return output

        parts = output.rsplit("\n\n", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        return output.strip()
        
    async def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """              
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        spatial_str = ""
        pruned_patch_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if self.role == 'Wiki Searcher' and info['role']=='Knowlegable Expert':
                queries = find_strings_between_pluses(info['output'])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            formatted_output = self._format_shared_output(info)
            if info.get("message_type") == "pruned_tail_patch":
                pruned_patch_str += f"Agent {id}, role is {info['role']}, final conclusion snippet is:\n\n {formatted_output}\n\n"
            else:
                spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {formatted_output}\n\n"
        for id, info in temporal_info.items():
            formatted_output = self._format_shared_output(info)
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {formatted_output}\n\n"
            
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"For pruned communication edges, only the final conclusion snippets of other agents are available:\n\n{pruned_patch_str}\n\n" if len(pruned_patch_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message, temperature=kwargs.get("temperature"))
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message, temperature=kwargs.get("temperature"))
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        return response
