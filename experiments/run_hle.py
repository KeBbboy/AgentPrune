import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_benchmark_suite import main


if __name__ == "__main__":
    asyncio.run(main(default_benchmark="hle"))
