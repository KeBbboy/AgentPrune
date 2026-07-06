import itertools
import re
from typing import Any, Dict, List, Union

from AgentDropout.prompt.common import get_combine_materials
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry


roles = itertools.cycle(
    [
        "Research Solver",
        "Evidence Analyst",
        "Computational Reasoner",
        "Skeptical Verifier",
    ]
)


ROLE_DESCRIPTION = {
    "Web Search Specialist": (
        "You specialize in external factual evidence. Use web_search when current, obscure, or source-grounded "
        "facts are needed, write focused queries, and avoid unsupported claims."
    ),
    "Code Execution Specialist": (
        "You specialize in computation and verification. Use code_generator for arithmetic, symbolic checks, "
        "small simulations, or data transformations when they can reduce reasoning errors."
    ),
    "Document Inspection Specialist": (
        "You specialize in attached files. Use text_inspector to inspect documents, tables, images with text, "
        "or other provided artifacts before answering file-dependent questions."
    ),
    "Independent Solver": (
        "You solve the problem directly from the question and other agents' outputs. Prefer clear reasoning "
        "and only rely on tool observations that are actually provided."
    ),
    "Synthesis Verifier": (
        "You compare candidate answers, check consistency across evidence and computations, and produce a "
        "careful final judgment."
    ),
    "Research Solver": (
        "You solve benchmark questions by combining the given problem, tool observations, "
        "and other agents' answers. Be concise and factual."
    ),
    "Evidence Analyst": (
        "You check whether evidence from search results or attachments supports the answer. "
        "Prefer directly grounded facts over speculation."
    ),
    "Computational Reasoner": (
        "You are strong at math, logic, and small computations. Use calculations when helpful."
    ),
    "Skeptical Verifier": (
        "You look for mistakes in assumptions, arithmetic, option matching, and unsupported claims."
    ),
    "Fake": "You are a liar who only tells lies.",
}


class BenchmarkPromptSet(PromptSet):
    answer_kind = "qa"
    choice_letters = "ABCD"

    @staticmethod
    def get_role() -> str:
        return next(roles)

    @classmethod
    def get_analyze_constraint(cls, role: str, use_structured_output: bool = False) -> str:
        role_instruction = ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Research Solver"])
        return (
            f"{role_instruction}\n"
            f"{cls._answer_format()}\n"
            "Use tool observations when they are present, but ignore irrelevant or contradictory snippets. "
            "When other agents provide outputs, use them as advice and verify independently."
        )

    @classmethod
    def get_constraint(cls, role: str | None = None) -> str:
        return cls.get_analyze_constraint(role or cls.get_role())

    @staticmethod
    def get_format() -> str:
        return "natural language"

    @staticmethod
    def get_answer_prompt(question: str, role: str = "") -> str:
        return f"Question:\n{question}"

    @staticmethod
    def get_adversarial_answer_prompt(question: str) -> str:
        return f"Give an incorrect answer and misleading reasoning for this question:\n{question}"

    @staticmethod
    def get_query_prompt(question: str) -> str:
        return f"Identify concise search queries or computations that would help answer:\n{question}"

    @staticmethod
    def get_file_analysis_prompt(query: str, file: str) -> str:
        return f"Question:\n{query}\n\nAttached file content:\n{file}"

    @staticmethod
    def get_websearch_prompt(query: str) -> str:
        return f"Search for facts needed to answer:\n{query}"

    @staticmethod
    def get_distill_websearch_prompt(query: str, results: Any) -> str:
        return f"Question:\n{query}\n\nSearch results:\n{results}\n\nSummarize only answer-relevant facts."

    @staticmethod
    def get_reflect_prompt(question: str, answer: str) -> str:
        return f"Check this answer for the question.\nQuestion:\n{question}\nAnswer:\n{answer}"

    @staticmethod
    def get_react_prompt(question: str, solutions: Any, feedback: Any) -> str:
        return f"Question:\n{question}\n\nPrior solution:\n{solutions}\n\nFeedback:\n{feedback}\n\nRevise the answer."

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_role() -> str:
        return "You are the final benchmark answer judge."

    @classmethod
    def get_decision_constraint(cls) -> str:
        return (
            "You will receive a benchmark question and outputs from other agents. "
            "Choose the most reliable final answer.\n"
            f"{cls._answer_format()}"
        )

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    @classmethod
    def postprocess_answer(cls, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            answer = str(answer)
        return cls._extract_answer(answer)

    @classmethod
    def _answer_format(cls) -> str:
        if cls.answer_kind == "multiple_choice":
            return (
                f"End with exactly one line: The answer is X, where X is one of {cls.choice_letters}."
            )
        if cls.answer_kind == "math":
            return "End with exactly one line: The answer is N, where N is the final integer."
        return "End with exactly one line: The answer is <short final answer>."

    @classmethod
    def _extract_answer(cls, answer: str) -> str:
        answer = answer.strip()
        if cls.answer_kind == "multiple_choice":
            match = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-J])\s*\)?", answer, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1).upper()
            match = re.search(r"\b([A-J])\b", answer)
            return match.group(1).upper() if match else ""
        if cls.answer_kind == "math":
            match = re.search(r"answer\s*(?:is|:)?\s*(-?\d+(?:\.\d+)?)", answer, flags=re.IGNORECASE)
            if match is None:
                match = re.search(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
            if match is None:
                return ""
            value = match.group(1) if len(match.groups()) else match.group(0)
            return value[:-2] if value.endswith(".0") else value
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            return match.group(1).strip()
        match = re.search(r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$", answer, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            return match.group(1).strip().rstrip(".")
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        return lines[-1].rstrip(".") if lines else ""


@PromptSetRegistry.register("gaia")
class GAIAPromptSet(BenchmarkPromptSet):
    answer_kind = "qa"


@PromptSetRegistry.register("hle")
class HLEPromptSet(BenchmarkPromptSet):
    answer_kind = "qa"


@PromptSetRegistry.register("musique")
class MuSiQuePromptSet(BenchmarkPromptSet):
    answer_kind = "qa"


@PromptSetRegistry.register("gpqa")
class GPQAPromptSet(BenchmarkPromptSet):
    answer_kind = "multiple_choice"
    choice_letters = "A, B, C, or D"


@PromptSetRegistry.register("aime")
class AIMEPromptSet(BenchmarkPromptSet):
    answer_kind = "math"

    @classmethod
    def get_analyze_constraint(cls, role: str, use_structured_output: bool = False) -> str:
        return (
            "You are a mathematical reasoning assistant. Solve the AIME problem accurately.\n"
            "Accuracy is more important than brevity. Work carefully through the reasoning, "
            "and verify numerical results whenever possible.\n"
            "Do not claim to use computation, search, external references, or tools unless an "
            "actual tool observation is provided in the conversation.\n"
            "Before giving the final answer, check arithmetic, edge cases, and whether the "
            "result satisfies the problem statement.\n"
            "For AIME, the final answer must be a non-negative integer from 0 to 999. "
            "If your computed answer is outside this range, re-check the interpretation and arithmetic.\n"
            f"{cls._answer_format()}"
        )

    @classmethod
    def get_decision_constraint(cls) -> str:
        return (
            "You will receive an AIME problem and outputs from other agents. "
            "Choose the most reliable final answer by checking the mathematical reasoning, "
            "arithmetic, and AIME range constraint.\n"
            "Do not introduce a new answer unless the provided reasoning is clearly wrong.\n"
            f"{cls._answer_format()}"
        )

    @classmethod
    def _answer_format(cls) -> str:
        return (
            "Provide the final answer in the format:\n"
            "\\boxed{YOUR_ANSWER}\n"
            "where YOUR_ANSWER is the final integer. Do not write anything after the boxed answer."
        )
