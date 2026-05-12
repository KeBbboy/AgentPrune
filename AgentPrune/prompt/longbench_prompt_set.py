from typing import Any, Dict, List, Union
import itertools
import re

from AgentPrune.prompt.common import get_combine_materials
from AgentPrune.prompt.prompt_set import PromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry


roles = itertools.cycle([
    "Evidence Extractor",
    "Long Context Analyst",
    "Answer Verifier",
    "Concise Responder",
])


ROLE_DESCRIPTION = {
    "Normal": (
        "You solve long-context tasks carefully and follow the task instruction exactly. "
        "Do not add formatting requirements that are not present in the task."
    ),
    "Evidence Extractor": (
        "You are good at finding the relevant evidence inside a long context. "
        "Follow the task instruction exactly and rely only on the provided context."
    ),
    "Long Context Analyst": (
        "You analyze long documents and connect scattered details. "
        "Follow the task instruction exactly and do not add extra explanation unless requested."
    ),
    "Answer Verifier": (
        "You verify candidate answers against the context. "
        "Prefer answers that are directly supported by the provided task and context."
    ),
    "Concise Responder": (
        "You produce concise final answers while following the task instruction exactly."
    ),
    "Fake": "You intentionally give a wrong answer and misleading reasoning.",
}


LONG_BENCH_BASE_CONSTRAINT = """
You will be given a LongBench task prompt. The task prompt already contains the
dataset-specific instruction. Follow that instruction exactly.
Do not change the requested answer style, output format, or level of explanation.
Use only the provided task/context.
"""


@PromptSetRegistry.register("longbench")
class LongBenchPromptSet(PromptSet):
    @staticmethod
    def get_role() -> str:
        return next(roles)

    @staticmethod
    def get_decision_role() -> str:
        return (
            "You are the final decision-maker for a LongBench task. "
            "Follow the task instruction exactly and choose the best supported answer."
        )

    @staticmethod
    def get_constraint() -> str:
        return LongBenchPromptSet.get_analyze_constraint("Normal")

    @staticmethod
    def get_analyze_constraint(role: str, use_structured_output: bool = False) -> str:
        role_instruction = ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Normal"])
        generic_instruction = LONG_BENCH_BASE_CONSTRAINT
        if role == "Fake":
            generic_instruction = """
You will be given a LongBench task prompt. Ignore the task intent and give an incorrect
answer with misleading reasoning.
"""
        return f"{role_instruction}\n{generic_instruction}"

    @staticmethod
    def get_decision_constraint() -> str:
        return """
You will receive a LongBench task prompt and several agents' outputs. Follow the
task instruction exactly. Choose the answer that is best supported by the provided
context and the strongest agent evidence.
"""

    @staticmethod
    def get_format() -> str:
        return LONG_BENCH_BASE_CONSTRAINT

    @staticmethod
    def get_answer_prompt(question: str, role: str = "") -> str:
        return question

    @staticmethod
    def get_query_prompt(question: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query: str, file: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question: str) -> str:
        return (
            "Give a wrong answer and false analysis for the following LongBench task prompt. "
            f"Task: {question}"
        )

    @staticmethod
    def get_distill_websearch_prompt(query: str, results: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question: str, answer: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if answer is None:
            return ""
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()
        answer = answer.split("\n\n", 1)[0].strip()
        answer = re.split(r"\n\s*Reasoning\s*[:：]", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        answer = re.sub(r"^\s*Answer\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
        return answer
