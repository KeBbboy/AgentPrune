from typing import Any, Dict, List, Union
import itertools
import re

from AgentPrune.prompt.prompt_set import PromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.prompt.common import get_combine_materials
from AgentPrune.prompt.mmlu_prompt_set import ROLE_DESCRIPTION


OPTION_LETTERS = "ABCDEFGHIJ"
OPTION_LIST_TEXT = ", ".join(OPTION_LETTERS)
OPTION_OR_TEXT = ", ".join(OPTION_LETTERS[:-1]) + f", or {OPTION_LETTERS[-1]}"

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
    """
    MMLU-Pro prompt set for multiple-choice question answering with up to 10 options.
    """

    @staticmethod
    def _strict_answer_format() -> str:
        return f"""
You must solve the multiple-choice question and select exactly one option: {OPTION_LIST_TEXT}.
Only output a letter that corresponds to an option actually shown in the question.
The first character of your response must be the final answer letter.
The first line of your response must contain only that single uppercase letter.
Do not put any words, punctuation, or spaces before the answer letter.
If you include any explanation, put it after the first line.
"""

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    @staticmethod
    def get_constraint():
        return f"""
            I will ask you a question.
            I will also give you multiple answers enumerated with letters from A to J.
            Only one answer out of the offered choices is correct.
            You must choose the correct answer to the question.
            Your response must be a single uppercase letter among the shown options.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : {OPTION_OR_TEXT})
        """ + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_analyze_constraint(role, use_structured_output: bool = False):
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        generic_instruction = f"""
I will ask you a question and multiple answers enumerated with letters from A to J.
Only one answer out of the offered choices is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents.
Your response must be a single uppercase letter among the shown options.
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : {OPTION_OR_TEXT})
"""
        role_prefix = f"{role_instruction}\n" if role_instruction else ""
        return role_prefix + generic_instruction + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_decision_constraint():
        return f"""
        I will ask you a question.
        I will also give you multiple answers enumerated with letters from A to J.
        Only one answer out of the offered choices is correct.
        You must choose the correct answer to the question.
        Your response must be a single uppercase letter among the shown options.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
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
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : {OPTION_OR_TEXT})
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

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) == 0:
            return answer

        xml_answer_match = re.search(r"<answer>\s*([A-J])\s*</answer>", answer, flags=re.IGNORECASE)
        if xml_answer_match is not None:
            return xml_answer_match.group(1).upper()

        answer_is_match = re.search(r"answer is\s*[:\-]?\s*([A-J])", answer, flags=re.IGNORECASE)
        if answer_is_match is not None:
            return answer_is_match.group(1).upper()

        for char in answer:
            upper_char = char.upper()
            if upper_char in OPTION_LETTERS:
                return upper_char
        return answer[:1].upper()
