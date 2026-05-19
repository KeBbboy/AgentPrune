from typing import Any, Dict, List, Union
import itertools
import re

from AgentPrune.prompt.common import get_combine_materials
from AgentPrune.prompt.prompt_set import PromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry


roles = itertools.cycle([
    "Bridge Reasoner",
    "Comparison Reasoner",
    "Evidence Verifier",
    "Context Synthesizer",
    "Answer Extractor",
])

ROLE_DESCRIPTION = {
    "Bridge Reasoner": (
        "You specialize in bridge-style multi-hop questions.\n"
        "Connect evidence across paragraphs to find the missing entity.\n\n"
        "First, check whether you have been assigned documents:\n"
        "IF you have documents:\n"
        "  -> Identify the two-hop reasoning chain from your documents.\n"
        "IF you have no documents:\n"
        "  -> Use messages from other agents to construct the bridge chain.\n"
        "  -> Do not fabricate evidence."
    ),
    "Comparison Reasoner": (
        "You specialize in comparison questions.\n"
        "Compare entities, quantities, dates, or attributes.\n\n"
        "First, check whether you have been assigned documents:\n"
        "IF you have documents:\n"
        "  -> Extract and compare relevant attributes from your documents.\n"
        "IF you have no documents:\n"
        "  -> Use messages from other agents as comparison evidence.\n"
        "  -> Do not fabricate evidence."
    ),
    "Evidence Verifier": (
        "You verify whether the proposed answer is supported by evidence.\n\n"
        "First, check whether you have been assigned documents:\n"
        "IF you have documents:\n"
        "  -> Verify directly against your documents.\n"
        "IF you have no documents:\n"
        "  -> Verify against messages from other agents.\n"
        "  -> Note: verification confidence is lower without original documents."
    ),
    "Context Synthesizer": (
        "You synthesize multiple pieces of context into one concise answer.\n\n"
        "First, check whether you have been assigned documents:\n"
        "IF you have documents:\n"
        "  -> Synthesize directly from your documents.\n"
        "IF you have no documents:\n"
        "  -> Synthesize from messages provided by other agents.\n"
        "  -> Do not fabricate evidence."
    ),
    "Answer Extractor": (
        "You extract the shortest answer span or yes/no decision.\n\n"
        "First, check whether you have been assigned documents:\n"
        "IF you have documents:\n"
        "  -> Extract the answer span directly from your documents.\n"
        "IF you have no documents:\n"
        "  -> Extract from the content shared by other agents.\n"
        "  -> Do not fabricate evidence."
    ),
}


@PromptSetRegistry.register("hotpotqa")
class HotpotQAPromptSet(PromptSet):
    @staticmethod
    def get_role() -> str:
        return next(roles)

    @staticmethod
    def get_constraint() -> str:
        return "Answer the HotpotQA question using your assigned documents and messages from other agents."

    @staticmethod
    def get_analyze_constraint(role: str, use_structured_output: bool = False) -> str:
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        return role_instruction if role_instruction else HotpotQAPromptSet.get_constraint()

    @staticmethod
    def get_format() -> str:
        return "short free-form answer"

    @staticmethod
    def get_answer_prompt(question: str) -> str:
        return question

    @staticmethod
    def get_decision_role() -> str:
        return (
            "You are the HotpotQA final decision maker. You consolidate other agents' "
            "answers and choose the answer best supported by the supplied context."
        )

    @staticmethod
    def get_decision_constraint() -> str:
        return (
            "Use the task context and the other agents' outputs to produce the final "
            "HotpotQA answer. Prefer answers that are directly supported by evidence in "
            "the context."
        )

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        answer = answer.strip()
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if answer_match is not None:
            return answer_match.group(1).strip()
        return answer

    @staticmethod
    def get_query_prompt(question: str) -> str:
        return question

    @staticmethod
    def get_file_analysis_prompt(query: str, file: str) -> str:
        return f"Question:\n{query}\n\nContext:\n{file}"

    @staticmethod
    def get_websearch_prompt(question: str) -> str:
        return question

    @staticmethod
    def get_adversarial_answer_prompt(question: str) -> str:
        return (
            "Give an unsupported or wrong HotpotQA answer for the following task, "
            "while keeping the response plausible:\n"
            f"{question}"
        )

    @staticmethod
    def get_distill_websearch_prompt(query: str, results: Any) -> str:
        return f"Question:\n{query}\n\nSearch results:\n{results}"

    @staticmethod
    def get_reflect_prompt(question: str, answer: str) -> str:
        return f"Question:\n{question}\n\nPrevious answer:\n{answer}"

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
