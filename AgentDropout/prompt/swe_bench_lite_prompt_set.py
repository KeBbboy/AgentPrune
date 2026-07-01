from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

roles = itertools.cycle(['Project Manager',
                         'Algorithm Designer',
                         'Programming Expert',
                         'Test Analyst',
                         'Bug Fixer',])

ROLE_DESCRIPTION = {
    "Project Manager": 
        "You are a project manager. "
        "You will be given a software engineering issue description. "
        "You are responsible for overseeing the overall structure of the fix, ensuring that the code changes are structured to complete the task. "
        "Implement code concisely and correctly without pursuing over-engineering."
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the fix, including which files need to be modified and the approach used.",
    "Algorithm Designer":
        "You are an algorithm designer. "
        "You will be given a software engineering issue description. "
        "You need to specify the specific design of the solution, including the algorithms and data structures to be used. "
        "You need to generate the detailed documentation, including explanations of the approach, usage instructions, and API references. "
        "When the implementation logic is complex, you can give the pseudocode logic of the main algorithm.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a software engineering issue description. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation as a unified diff patch (git diff format). "
        "Use a code block to write your response. For example:\n```\ndiff --git a/file.py b/file.py\nindex xxx..yyy\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old_code\n+new_code\n```\nDo not include anything other than the patch in your response.",
    "Test Analyst":
        "You are a test analyst. "
        "You will be given a software engineering issue description. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code.",
    "Bug Fixer":
        "You are a bug fixer."
        "You will be given a software engineering issue description. "
        "You need to provide modified and improved code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation as a unified diff patch (git diff format). "
        "Use a code block to write your response. For example:\n```\ndiff --git a/file.py b/file.py\nindex xxx..yyy\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old_code\n+new_code\n```\nDo not include anything other than the patch in your response",
    "Normal Programmer":
        "You are a programmer. "
        "You will be given a software engineering issue description. "
        "You can refer to the agents' outputs. "
        "Write your full implementation as a unified diff patch (git diff format). "
        "Use a code block to write your response. For example:\n```\ndiff --git a/file.py b/file.py\nindex xxx..yyy\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old_code\n+new_code\n```\nDo not include anything other than the patch in your response.",
    "Stupid Programmer":
        "You are a stupid programmer. "
        "You will be given a software engineering issue description. "
        "Give a code implementation full of errors. "
        "Do not use comments for all errors. "
        "Use a code block to write your response. For example:\n```\ndiff --git a/file.py b/file.py\nindex xxx..yyy\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old_code\n+new_code\n```\nDo not include anything other than the patch in your response.",
}


@PromptSetRegistry.register('swe_bench_lite')
class SWEBenchLitePromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question):
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following issue:
Issue:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following issue:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Issue Resolution\n\n"
"Evaluate if additional information is needed to fix the issue. "
"If file analysis is required, formulate specific queries to assist in finding the relevant code.\n\n"
f"## ❓ Target Issue:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the issue that are essential for finding the solution.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Issue: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original issue. Each query should focus on key terms from the issue. Format the output as a comma-separated list.\n"
            "For example, if the issue is 'How to fix memory leak in Python async code?', your queries could be: 'Python async memory leak, asyncio cleanup, Python garbage collection'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )


    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
"# Summarization of Search Results\n\n"
f"## Original issue: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the issue and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the issue.\n"
"2. Compare the answers for their adherence to the given issue's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the issue's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_constraint():
        return (
"You will be given a software engineering issue description."
"You may be given the overall code design, algorithm framework, code implementation or test problems."
"Write your full implementation as a unified diff patch (git diff format). "
"If the prompt given to you contains code that passed internal testing, you can choose the most reliable reply."
"If there is no code that has passed internal testing in the prompt, you can change it yourself according to the prompt."
"Use a code block to write your response. For example:\n```\ndiff --git a/file.py b/file.py\nindex xxx..yyy\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old_code\n+new_code\n```\nDo not include anything other than the patch in your response"
)
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers. And you are an AI that only responds with code patches."
    
    @staticmethod
    def get_decision_few_shot():
        return ""