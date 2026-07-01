from typing import Any, Dict, List, Union
import itertools

from AgentDropout.prompt.common import get_combine_materials
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry


roles = itertools.cycle([
    "Context Analyst",
    "Evidence Checker",
    "Task Solver",
    "Skeptical Reviewer",
])


ROLE_DESCRIPTION = {
    "Context Analyst": (
        "You are a context-learning analyst. Extract the task-relevant facts from the provided context, "
        "then answer using only those facts."
    ),
    "Evidence Checker": (
        "You are an evidence checker. Verify each claim against the provided context and avoid unsupported assumptions."
    ),
    "Task Solver": (
        "You are a task solver. Follow the task exactly and produce a concise answer grounded in the context."
    ),
    "Skeptical Reviewer": (
        "You are a skeptical reviewer. Look for ambiguity, conflicts, and missing information before answering."
    ),
    "Fake": "You are a liar who only tells lies.",
}


@PromptSetRegistry.register("cl_bench")
class CLBenchPromptSet(PromptSet):
    @staticmethod
    def get_role() -> str:
        return next(roles)

    @staticmethod
    def get_decision_role() -> str:
        return "You are the final decision-maker for a context-learning benchmark."

    @staticmethod
    def get_constraint() -> str:
        return CLBenchPromptSet.get_analyze_constraint(CLBenchPromptSet.get_role())

    @staticmethod
    def get_analyze_constraint(role: str, use_structured_output: bool = False) -> str:
        role_instruction = ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Task Solver"])
        return (
            f"{role_instruction}\n"
            "You will receive a context and a task. Answer the task using only information in the context.\n"
            "If other agents provide reasoning, use it critically and correct unsupported claims.\n"
            "Put the final answer first. If useful, add a short explanation after it."
        )

    @staticmethod
    def get_decision_constraint() -> str:
        return (
            "You will receive a context-learning task and answers from other agents. "
            "Choose the answer best supported by the provided context. "
            "Return the final answer first, without adding unsupported information."
        )

    @staticmethod
    def get_format() -> str:
        return "natural language"

    @staticmethod
    def get_answer_prompt(question: str) -> str:
        return f"{question}"

    @staticmethod
    def get_adversarial_answer_prompt(question: str) -> str:
        return (
            "Give an incorrect answer for the following context-learning task and provide misleading reasoning:\n"
            f"{question}"
        )

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
    def get_distill_websearch_prompt(query: str, results: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question: str, answer: str) -> str:
        raise NotImplementedError

    @staticmethod
    def get_react_prompt(question: str, solutions: Any, feedback: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            answer = str(answer)
        answer = answer.strip()
        for prefix in ("Final answer:", "Answer:", "The answer is"):
            if answer.lower().startswith(prefix.lower()):
                return answer[len(prefix):].strip()
        return answer
