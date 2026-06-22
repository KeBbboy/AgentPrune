from typing import Union, Dict, Any, List
import itertools
import re

from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials


roles = itertools.cycle(['Knowlegable Expert',
                        #  'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'])


ROLE_DESCRIPTION = {
"Knowlegable Expert":
"""
You are a knowlegable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem. 
Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
# "Knowlegable Expert":
# """
# You are a knowlegable expert in question answering.
# Please answer the question.
# """,
# "Knowlegable Expert":
# """
# You are a knowlegable expert in question answering.
# Please think step by step and answer the question.
# """,
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
"Doctor":
"""
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
"Lawyer":
"""
You are good at law, politics, and history.
""",
"Economist":
"""
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
"Programmer":
"""
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
"Fake":
"""
You are a liar who only tell lies.
""",
}


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def _strict_answer_format() -> str:
        return """
                You must solve the multiple-choice question and select exactly one option: A, B, C, or D.
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
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """+ MMLUPromptSet._strict_answer_format()
    
    @staticmethod
#     def get_analyze_constraint(role):
#         return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else """
# I will ask you a question and 4 answers enumerated as A, B, C and D.
# Only one answer out of the offered 4 is correct.
# Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
# You are strictly prohibited from imitating the analysis process of other agents
# Please think step by step.
# Your reply must include your answer and a brief step by step analysis of the question.
# The first line of your reply must contain only one letter(for example : A, B, C or D)
# """
    @staticmethod
    def get_analyze_constraint(role, use_structured_output: bool = False):
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        generic_instruction = """
            I will ask you a question and 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
            You are strictly prohibited from imitating the analysis process of other agents
            Your reply must include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
            """
        role_prefix = f"{role_instruction}\n" if role_instruction else ""
        return role_prefix + generic_instruction + MMLUPromptSet._strict_answer_format()
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """ + MMLUPromptSet._strict_answer_format()
    
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
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """
    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
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

    @staticmethod
    def _extract_option_answer(answer: str, valid_letters: str = "ABCD") -> str:
        escaped_letters = re.escape(valid_letters)
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if not lines:
            return ""

        for pattern in (
            rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\*\*)?\b",
            rf"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer|correct\s+answer)\s*(?:is|:)?\s*(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\*\*)?\b",
        ):
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1).upper()

        first_line_match = re.match(
            rf"^(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\s*(?:\*\*)?)?(?:[\s\).:：-]|$)",
            lines[0],
            flags=re.IGNORECASE,
        )
        if first_line_match is not None:
            return first_line_match.group(1).upper()

        option_mentions = {
            match.group(1).upper()
            for match in re.finditer(
                rf"\boption\s+([{escaped_letters}])\b",
                answer,
                flags=re.IGNORECASE,
            )
        }
        if len(option_mentions) == 1:
            return next(iter(option_mentions))
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        return self._extract_option_answer(answer)


@PromptSetRegistry.register('mmlu_pro')
class MMLUProPromptSet(MMLUPromptSet):
    """
    MMLU-Pro prompt set for variable-length multiple-choice questions.
    """

    @staticmethod
    def _strict_answer_format() -> str:
        return """
                You must solve the multiple-choice question and select exactly one option from A through J.
                The first character of your response must be the final answer letter.
                The first line of your response must contain only that single uppercase letter.
                Do not put any words, punctuation, or spaces before the answer letter.
                If you include any explanation, put it after the first line.
                """

    @staticmethod
    def get_constraint():
        return """
            I will ask you a multiple-choice question.
            You must choose the correct option from the options shown in the question.
            Your response must be one of the letters A through J, corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter.
        """ + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_analyze_constraint(role, use_structured_output: bool = False):
        role_instruction = ROLE_DESCRIPTION.get(role, "")
        generic_instruction = """
            I will ask you a multiple-choice question.
            Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
            You are strictly prohibited from imitating the analysis process of other agents.
            Your reply must include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter from A through J.
            """
        role_prefix = f"{role_instruction}\n" if role_instruction else ""
        return role_prefix + generic_instruction + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a multiple-choice question.
        You must choose the correct option from the options shown in the question.
        Your response must be one of the letters A through J, corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """ + MMLUProPromptSet._strict_answer_format()

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                The first line of your reply must contain only one letter from A through J.
                """
