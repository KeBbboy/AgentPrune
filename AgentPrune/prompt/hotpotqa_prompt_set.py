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
        "You specialize in bridge-style multi-hop questions. Connect evidence across "
        "paragraphs and identify the missing entity or fact."
    ),
    "Comparison Reasoner": (
        "You specialize in comparison questions. Compare the relevant entities, "
        "quantities, dates, or attributes using only the provided context."
    ),
    "Evidence Verifier": (
        "You verify candidate reasoning against the context and reject unsupported claims."
    ),
    "Context Synthesizer": (
        "You synthesize multiple context paragraphs into one concise answer."
    ),
    "Answer Extractor": (
        "You extract the shortest answer span or yes/no decision supported by the context."
    ),
}

HOTPOTQA_OUTPUT_CONSTRAINT = """
Use only the supplied HotpotQA context. The official answer is a short free-form
answer string, usually an entity, phrase, number, date, yes, no, or noanswer.
Reason briefly when useful, but the final line must be exactly:
<answer>your short answer</answer>
Do not include supporting-fact titles, sentence ids, citations, or explanations inside the answer tag.
"""


@PromptSetRegistry.register("hotpotqa")
class HotpotQAPromptSet(PromptSet):
    @staticmethod
    def get_role() -> str:
        return next(roles)

    @staticmethod
    def get_constraint() -> str:
        return HOTPOTQA_OUTPUT_CONSTRAINT

    @staticmethod
    def get_analyze_constraint(role: str, use_structured_output: bool = False) -> str:
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        if role_instruction:
            return role_instruction + "\n" + HOTPOTQA_OUTPUT_CONSTRAINT
        return HOTPOTQA_OUTPUT_CONSTRAINT

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
            "the context. The final line must be exactly:\n"
            "<answer>your short answer</answer>\n"
            "Do not include explanations inside the answer tag."
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
            "while still using the required answer tag format:\n"
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
