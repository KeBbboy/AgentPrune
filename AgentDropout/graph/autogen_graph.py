from typing import Any, Dict, Optional

from AgentDropout.graph.graph import Graph


class GraphAutoGen(Graph):
    async def arun(
        self,
        input: Dict[str, str],
        num_rounds: int = 3,
        max_tries: int = 3,
        max_time: int = 6000,
        aggregate_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        return await super().arun(
            input=input,
            num_rounds=num_rounds,
            max_tries=max_tries,
            max_time=max_time,
            **kwargs,
        )
