from importlib import import_module

from AgentDropout.agents.agent_registry import AgentRegistry


_AGENT_MODULES = {
    "AnalyzeAgent": "AgentDropout.agents.analyze_agent",
    "AdverarialAgent": "AgentDropout.agents.adversarial_agent",
    "CodeWriting": "AgentDropout.agents.code_writing",
    "CodeWritingAG": "AgentDropout.agents.code_writing",
    "MathSolver": "AgentDropout.agents.math_solver",
    "MathSolverAG": "AgentDropout.agents.math_solver",
    "MathSolver_aqua": "AgentDropout.agents.math_solver_aqua",
    "FinalDirect": "AgentDropout.agents.final_decision",
    "FinalMajorVote": "AgentDropout.agents.final_decision",
    "FinalRefer": "AgentDropout.agents.final_decision",
    "FinalReferAG": "AgentDropout.agents.final_decision",
    "FinalWriteCode": "AgentDropout.agents.final_decision",
    "FinalWriteCodeAG": "AgentDropout.agents.final_decision",
}
_LOADED_MODULES = set()


def ensure_agent_modules_loaded(agent_names):
    for agent_name in agent_names:
        module_name = _AGENT_MODULES.get(agent_name)
        if module_name is None or module_name in _LOADED_MODULES:
            continue
        import_module(module_name)
        _LOADED_MODULES.add(module_name)


__all__ = ["AgentRegistry", "ensure_agent_modules_loaded"]
