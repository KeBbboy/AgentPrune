import sys
import os
import asyncio
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from dataset.svamp_dataset import SVAMPDataset
from experiments.run_math_dataset import add_shared_math_args, run_math_dataset_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune experiments on SVAMP")
    add_shared_math_args(
        parser,
        default_dataset_dir="dataset/svamp",
        default_dataset_json="dataset/svamp/test.json",
        default_domain="svamp",
    )
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    await run_math_dataset_experiment(args, SVAMPDataset)


if __name__ == "__main__":
    asyncio.run(main())
