from typing import Union, Dict, Any, List
import itertools
import re

from AgentPrune.prompt.prompt_set import PromptSet
from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.prompt.common import get_combine_materials
from AgentPrune.prompt.mmlu_prompt_set import ROLE_DESCRIPTION


roles = itertools.cycle([
    'Knowlegable Expert',
    'Critic',
    'Mathematician',
    'Psychologist',
    'Historian',
    'Doctor',
    'Lawyer',
    'Economist',
    'Programmer',
])


AQUA_XML_ANALYZE_FORMAT_CONSTRAINT = """
Answer the multiple-choice question by analyzing it from your role's perspective.

CRITICAL OUTPUT FORMAT — you MUST respond with exactly these three XML blocks,
in this order, with no text before, between, or after them:

<claim>One sentence stating your core position.</claim>
<steps>
- First key reasoning point
- Second key reasoning point
- Additional points as needed (no upper limit, but each should be one short bullet)
</steps>
<answer>A</answer>

The <answer> tag must contain ONLY a single letter from {A, B, C, D, E}.
Do not include explanations inside <answer>.
Do not output any text outside these three blocks.
"""


@PromptSetRegistry.register('aqua')
class AQuAPromptSet(PromptSet):
    @staticmethod
    def _strict_answer_format() -> str:
        return """
You must solve the multiple-choice question and select exactly one option: A, B, C, D, or E.
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
        return """
            I will ask you a question.
            I will also give you 5 answers enumerated as A, B, C, D and E.
            Only one answer out of the offered 5 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 5 letters: A, B, C, D or E,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C, D or E)
        """ + AQuAPromptSet._strict_answer_format()

    @staticmethod
    def get_analyze_constraint(role, use_structured_output: bool = False):
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        generic_instruction = """
I will ask you a question and 5 answers enumerated as A, B, C, D and E.
Only one answer out of the offered 5 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents.
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C, D or E)
"""
        if use_structured_output:
            return f"{role_instruction}\n{generic_instruction}\n{AQUA_XML_ANALYZE_FORMAT_CONSTRAINT}"
        role_prefix = f"{role_instruction}\n" if role_instruction else ""
        return role_prefix + generic_instruction + AQuAPromptSet._strict_answer_format()

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 5 answers enumerated as A, B, C, D and E.
        Only one answer out of the offered 5 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 5 letters: A, B, C, D or E,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """ + AQuAPromptSet._strict_answer_format()

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
                The first line of your reply must contain only one letter(for example : A, B, C, D or E)
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

        xml_answer_match = re.search(r"<answer>\s*([ABCDE])\s*</answer>", answer, flags=re.IGNORECASE)
        if xml_answer_match is not None:
            return xml_answer_match.group(1).upper()

        answer_is_match = re.search(r"answer is\s*[:\-]?\s*([ABCDE])", answer, flags=re.IGNORECASE)
        if answer_is_match is not None:
            return answer_is_match.group(1).upper()

        for char in answer:
            upper_char = char.upper()
            if upper_char in {"A", "B", "C", "D", "E"}:
                return upper_char
        return answer[:1].upper()
