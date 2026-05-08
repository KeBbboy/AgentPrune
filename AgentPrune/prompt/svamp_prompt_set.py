from AgentPrune.prompt.gsm8k_prompt_set import GSM8KPromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register("svamp")
class SVAMPPromptSet(GSM8KPromptSet):
    pass
