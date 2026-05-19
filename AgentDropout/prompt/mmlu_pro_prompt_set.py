from typing import Any, Dict, List, Union
import itertools
import random
import re

from AgentDropout.prompt.common import get_combine_materials
from AgentDropout.prompt.mmlu_prompt_set import ROLE_DESCRIPTION
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry


OPTION_LETTERS = "ABCDEFGHIJ"
OPTION_LIST_TEXT = ", ".join(OPTION_LETTERS)

roles = itertools.cycle([
    "Knowlegable Expert",
    "Critic",
    "Mathematician",
    "Psychologist",
    "Historian",
    "Doctor",
    "Lawyer",
    "Economist",
    "Programmer",
])


@PromptSetRegistry.register("mmlu_pro")
class MMLUProPromptSet(PromptSet):
    """MMLU-Pro prompt set for multiple-choice QA with up to 10 options."""

    @staticmethod
    def _strict_answer_format() -> str:
        return f"""
You must solve the multiple-choice question and select exactly one option: {OPTION_LIST_TEXT}.
Only output a letter that corresponds to an option actually shown in the question.
Reason step by step, and end your response with: The answer is (X)
"""

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    @staticmethod
    def get_constraint():
        return """
            I will give you a multiple-choice question with options labeled from A to J.
            Only one answer is correct.
            Think step by step and then finish with the exact format: The answer is (X)
            Your answer can refer to the answers of other agents provided to you.
        """ + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_analyze_constraint(role, use_structured_output: bool = False):
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        generic_instruction = """
I will give you a multiple-choice question with options labeled from A to J.
Only one answer is correct.
Using the reasoning from other agents as additional advice with critical thinking, give an updated answer.
You are strictly prohibited from imitating the analysis process of other agents.
Think step by step and then finish with the exact format: The answer is (X)
"""
        role_prefix = f"{role_instruction}\n" if role_instruction else ""
        return role_prefix + generic_instruction + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_decision_constraint():
        return """
        I will give you a multiple-choice question with options labeled from A to J.
        Only one answer is correct.
        I will also give you other people's answers and analysis.
        Think step by step and then finish with the exact format: The answer is (X)
        """ + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Think step by step and end with the exact format: The answer is (X)
                """

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot():
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]], valid_options: List[str] | None = None) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return answer

        valid_letters = valid_options if valid_options else list(OPTION_LETTERS)
        valid_letter_group = "".join(valid_letters)

        answer_is_match = re.search(
            rf"answer is\s*\(?([{valid_letter_group}])\)?",
            answer,
            flags=re.IGNORECASE,
        )
        if answer_is_match is not None:
            return answer_is_match.group(1).upper()

        answer_colon_match = re.search(
            rf"answer:\s*\(?([{valid_letter_group}])\)?",
            answer,
            flags=re.IGNORECASE,
        )
        if answer_colon_match is not None:
            return answer_colon_match.group(1).upper()

        standalone_matches = re.findall(
            rf"\b([{valid_letter_group}])\b",
            answer,
            flags=re.IGNORECASE,
        )
        if standalone_matches:
            return standalone_matches[-1].upper()

        return random.Random(answer).choice(valid_letters)
