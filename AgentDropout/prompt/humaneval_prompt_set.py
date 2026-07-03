from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

roles = itertools.cycle(['Project Manager',
                         'Algorithm Designer',
                         'Programming Expert',
                         'Test Analyst',
                         'Bug Fixer',
                         'API Designer',
                         'Edge-Case Thinker',
                         'Performance Optimizer',
                         'Refactoring Specialist',
                         'Documentation Writer',
                         'Type Hint Specialist',
                         'Test Case Generator',
                         'Security Auditor',
                         'Pythonic Coder',
                         'Idiomatic Implementer',])

ROLE_DESCRIPTION = {
    "Project Manager": 
        "You are a project manager. "
        "You will be given a function signature and its docstring by the user. "
        "You are responsible for overseeing the overall structure of the code, ensuring that the code is structured to complete the task Implement code concisely and correctly without pursuing over-engineering."
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the code, including the classes that need to be defined(maybe none) and the functions used (maybe only one function).",
    "Algorithm Designer":
        "You are an algorithm designer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to specify the specific design of the algorithm, including the classes that may be defined and the functions used. "
        "You need to generate the detailed documentation, including explanations of the algorithm, usage instructions, and API references. "
        "When the implementation logic is complex, you can give the pseudocode logic of the main algorithm.",
    # "Programming Expert":
    #     "You are a programming expert. "
    #     "You will be given a function signature and its docstring by the user. "
    #     "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
    #     "Write your full implementation (restate the function signature). "
    #     "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
    #     "Do not include anything other than Python code blocks in your response. "
    #     "Do not change function names and input variable types in tasks.",
    # "Programming Expert":
    #     "You are a programming expert. "
    #     "Please think step by step.",
    # "Programming Expert":
    #     "Do not include anything other than Python code blocks in your response. ",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks."
        "Please think step by step.",
    "Test Analyst":
        "You are a test analyst. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code.",
    "Bug Fixer":
        "You are a bug fixer."
        "You will be given a function signature and its docstring by the user. "
        "You need to provide modified and improved python code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response "
        "Do not change function names and input variable types in tasks",
    "Normal Programmer":
        "You are a programmer. "
        "You will be given a function signature and its docstring by the user. "
        "You can refer to the agents' outputs. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. ",
    "Stupid Programmer":
        "You are a stupid programmer. "
        "You will be given a function signature and its docstring by the user. "
        "Give a code implementation full of errors. "
        "Do not use comments for all errors. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. ",
    "API Designer":
        "You are an API designer. "
        "You will be given a function signature and its docstring by the user. "
        "Focus on the contract: parameter types, return type, error conditions, and side effects. "
        "Propose a clean public interface even if the original signature is loose. "
        "You may output a Python code block with a typed function signature (no implementation body required) for the rest of the agents to refine. "
        "Do not change function names and input variable types in tasks.",
    "Edge-Case Thinker":
        "You are an edge-case thinker. "
        "You will be given a function signature and its docstring by the user. "
        "Enumerate all edge cases: empty inputs, single-element inputs, very large inputs, negative numbers, None, type mismatches, division by zero, off-by-one indices. "
        "Propose how the implementation should behave on each case. "
        "Do not include code unless it is a tiny example illustrating a single edge case. "
        "Do not change function names and input variable types in tasks.",
    "Performance Optimizer":
        "You are a performance optimizer. "
        "You will be given a function signature and its docstring by the user. "
        "Analyze the time and space complexity implied by naive implementations and propose the optimal approach. "
        "When the function is amenable to dynamic programming, two pointers, hashing, or a built-in, name the technique explicitly. "
        "You may output a Python code block with a high-performance implementation (restate the function signature). "
        "Do not change function names and input variable types in tasks.",
    "Refactoring Specialist":
        "You are a refactoring specialist. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be clean. "
        "Refactor their code for readability: rename variables, extract helpers, reduce nesting, and remove dead branches. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
    "Documentation Writer":
        "You are a documentation writer. "
        "You will be given a function signature and its docstring by the user. "
        "Expand the docstring into a full description: purpose, parameters (with types and meaning), return value, raised exceptions, and a short usage example. "
        "Then provide a minimal reference implementation (restate the function signature) that satisfies the docstring. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
    "Type Hint Specialist":
        "You are a type hint specialist. "
        "You will be given a function signature and its docstring by the user. "
        "Add comprehensive type annotations (typing, typing.Optional, typing.Iterable, etc.) to every parameter and return. "
        "When generics or Protocols are useful, apply them. "
        "Then provide the implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
    "Test Case Generator":
        "You are a test case generator. "
        "You will be given a function signature and its docstring by the user. "
        "Write a self-contained set of assertions covering: the canonical case, a couple of edge cases, an empty case, and a stress case. "
        "Output the assertions inside a Python code block; the rest of the agents can use them to verify their implementations. "
        "Do not change function names and input variable types in tasks.",
    "Security Auditor":
        "You are a security auditor. "
        "You will be given a function signature and its docstring by the user. "
        "Identify potential security issues: untrusted input, eval/exec, subprocess, file system access, deserialization, integer overflow, and information leaks. "
        "Propose mitigations and, if possible, a safe implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
    "Pythonic Coder":
        "You are a Pythonic coder. "
        "You will be given a function signature and its docstring by the user. "
        "Write the most Pythonic solution: prefer built-ins (collections.Counter, itertools, functools, math), comprehensions, context managers, and unpacking. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
    "Idiomatic Implementer":
        "You are an idiomatic implementer. "
        "You will be given a function signature and its docstring by the user. "
        "Follow PEP 8 closely: snake_case, type hints, docstring format, and meaningful names. "
        "You may be able to get the output results of other agents. They may have passed internal tests but read awkwardly. "
        "Rewrite for clarity and idiomatic style. "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not change function names and input variable types in tasks.",
}


@PromptSetRegistry.register('humaneval')
class HumanEvalPromptSet(PromptSet):

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
        # Format the question for the AI assistant to answer
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
# "## 🤔 Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            # "# File Analysis Required\n\n"
            # f"## 🔍 Required Information to Extract:\n---\n{query}\n---\n\n"
            # f"## 📄 File Content for Analysis:\n---\n{file}\n---\n\n"
            # "## 🤔 Instructions:\n"
            # "Extract the specified information from the file. Example: 'Identify the main theme in the text.'"
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            # "# Summarization of Search Results\n\n"
            # "## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
            # "## 🌐 Search Results for Analysis:\n---\n{results}\n---\n\n"
            # "## ✏️ Instructions:\n"
            # "Summarize the key findings from the search results related to the query. "
            # "Focus on relevant information. Example: 'Summary of key points...'"
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
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
            # "# Self-Consistency Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given answers and choose the most consistent one. "
            # "If all answers differ, select the one you find most reliable. "
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Best Answer Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given question and candidate answers and choose the most reasonable one. "
            # "Please copy the original answer if you decide."
            # f"Please keep following the constraints to answer the question: {constraint}."
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
"You will be given a function signature and its docstring by the user."
"You may be given the overall code design, algorithm framework, code implementation or test problems."
"Write your full implementation (restate the function signature). "
"If the prompt given to you contains code that passed internal testing, you can choose the most reliable reply."
"If there is no code that has passed internal testing in the prompt, you can change it yourself according to the prompt."
"Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
"Do not include anything other than Python code blocks in your response"
)
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers. And you are an AI that only responds with only python code."
    
    @staticmethod
    def get_decision_few_shot():
        return ""
