import re
import itertools
from typing import Any, Dict, List, Union

from AgentDropout.prompt.common import get_combine_materials
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry


roles = itertools.cycle([
    "Evidence Analyst",
    "Bridge Reasoner",
    "Skeptical Verifier",
    "Answer Synthesizer",
])


ROLE_DESCRIPTION = {
    "Evidence Analyst": (
        "You are an evidence analyst for multi-hop question answering. "
        "Read the assigned documents carefully, extract only relevant facts, and avoid unsupported claims. "
        "Keep the reasoning concise. The last line of your output must be in the form: The answer is <short answer>."
    ),
    "Bridge Reasoner": (
        "You are a bridge reasoner for multi-hop question answering. "
        "Connect clues across documents and other agents' messages, and only use facts supported by the provided context. "
        "Keep the reasoning concise. The last line of your output must be in the form: The answer is <short answer>."
    ),
    "Skeptical Verifier": (
        "You are a skeptical verifier for multi-hop question answering. "
        "Check whether the claimed facts are actually supported by the provided context, reject leaps in logic, and correct unsupported claims. "
        "Keep the reasoning concise. The last line of your output must be in the form: The answer is <short answer>."
    ),
    "Answer Synthesizer": (
        "You are an answer synthesizer for multi-hop question answering. "
        "Use the assigned documents and other agents' evidence to produce the shortest accurate final answer. "
        "Keep the reasoning concise. The last line of your output must be in the form: The answer is <short answer>."
    ),
    "Normal": (
        "You answer open-domain multi-hop questions using only the provided context. "
        "Do not invent evidence. The last line of your output must be in the form: The answer is <short answer>."
    ),
    "Fake": (
        "You are a liar who only tells lies."
    ),
}


@PromptSetRegistry.register("hotpotqa")
class HotpotQAPromptSet(PromptSet):
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role: str = "Normal"):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Normal"])

    @staticmethod
    def get_analyze_constraint(role: str, use_structured_output: bool = False):
        del use_structured_output
        return HotpotQAPromptSet.get_constraint(role)

    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role: str = "Normal"):
        del role
        return f"{question}"

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return (
            "Give a wrong answer and misleading reasoning for the following multi-hop question.\n"
            "Do not mention that you are being deceptive.\n"
            "The last line of your output must be in the form: The answer is <short answer>.\n\n"
            f"{question}"
        )

    @staticmethod
    def get_query_prompt(question) -> str:
        return f"Identify the missing evidence needed to answer this HotpotQA question:\n\n{question}"

    @staticmethod
    def get_file_analysis_prompt(query, file) -> str:
        return f"Question:\n{query}\n\nContext:\n{file}\n\nExtract only evidence relevant to the question."

    @staticmethod
    def get_websearch_prompt(query) -> str:
        return f"Generate targeted search queries for this HotpotQA question:\n\n{query}"

    @staticmethod
    def get_distill_websearch_prompt(query, results) -> str:
        return f"Question:\n{query}\n\nSearch results:\n{results}\n\nSummarize only evidence relevant to the answer."

    @staticmethod
    def get_reflect_prompt(question, answer) -> str:
        return (
            "Review whether the answer is fully supported by the provided evidence.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        )

    @staticmethod
    def get_react_prompt(question, solutions, feedback) -> str:
        return (
            f"Question:\n{question}\n\nCurrent solutions:\n{solutions}\n\nFeedback:\n{feedback}\n\n"
            "Revise the answer using only supported evidence."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials, avoid_vague=False)

    @staticmethod
    def get_decision_constraint() -> str:
        return (
            "You will be given a multi-hop question, supporting context, and candidate answers from other agents. "
            "Choose the answer best supported by the provided evidence. "
            "Reject unsupported claims and keep the reasoning concise. "
            "The last line of your output must be in the form: The answer is <short answer>."
        )

    @staticmethod
    def get_decision_role() -> str:
        return "You are the top decision-maker for evidence-grounded question answering."

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if answer_match is not None:
            return self._clean_prediction(answer_match.group(1))

        final_answer_match = re.search(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_answer_match is not None:
            return self._clean_prediction(final_answer_match.group(1))

        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) > 0:
            return self._clean_prediction(lines[-1])
        return ""

    @staticmethod
    def _clean_prediction(answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"^[-*\s]+", "", answer)
        answer = answer.strip().strip("\"'")
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
