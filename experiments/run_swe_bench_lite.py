import sys
import os
import argparse
import json
import time
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import List, Union, Literal
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.graph.graph import Graph
from AgentDropout.llm.gpt_chat import configure_thinking, parse_bool_arg
from dataset.swe_bench_lite_dataset import SWEBenchLiteDataset
from AgentDropout.utils.globals import Time
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('swe_bench_lite.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on SWE-bench Lite (Official Evaluation)")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Directory to save results (default: auto-generated)")
    parser.add_argument("--llm_name", type=str, default="MiniMax-M3",
                        help="LLM model name to use")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['direct', 'fullconnect'],
                        help="Evaluation mode: 'direct' for single-agent evaluation, "
                             "'fullconnect' for multi-agent full-connected evaluation")
    parser.add_argument('--agent_nums', type=int, default=5,
                        help="Number of agents for fullconnect mode")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of inference rounds for multi-agent mode")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument('--domain', type=str, default="swe_bench_lite",
                        help="Domain name (default: swe_bench_lite)")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decision method of the agentprune')
    parser.add_argument('--max_workers', type=int, default=4,
                        help="Number of parallel workers for evaluation")
    parser.add_argument('--run_id', type=str, default=None,
                        help="Unique run identifier for evaluation")
    parser.add_argument('--skip_evaluation', action='store_true',
                        help="Only generate predictions, skip official evaluation")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode with verbose output")
    parser.add_argument('--llm_enable_thinking', type=parse_bool_arg, default=True,
                        help='Enable model thinking mode: true or false')
    parser.add_argument('--llm_thinking_budget', type=int, default=8192,
                        help='Thinking token budget for both Qwen3 and MiniMax-M3 providers')

    args = parser.parse_args()

    if args.mode == 'direct' and args.agent_nums != 5:
        logger.warning(f"Direct mode ignores agent_nums={args.agent_nums}, using single agent")

    if args.run_id is None:
        args.run_id = f"agentprune_{args.mode}_{args.llm_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    return args


def get_kwargs(mode: Union[Literal['direct'], Literal['fullconnect']], N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    if mode == 'direct':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]

    elif mode == 'fullconnect':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


def run_official_evaluation(predictions_path, dataset_name, run_id, max_workers=4):
    logger.info(f"=" * 80)
    logger.info(f"Running official SWE-bench evaluation")
    logger.info(f"=" * 80)
    logger.info(f"Predictions path: {predictions_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"=" * 80)

    cmd = [
        sys.executable, '-m', 'swebench.harness.run_evaluation',
        '--dataset_name', dataset_name,
        '--predictions_path', str(predictions_path),
        '--max_workers', str(max_workers),
        '--run_id', run_id,
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd()
    )

    for line in process.stdout:
        logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        logger.error(f"Official evaluation failed with return code: {process.returncode}")
        return None

    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        logger.warning("Evaluation results directory not found")
        return None

    return results_dir


def parse_official_results(results_dir):
    if not results_dir or not results_dir.exists():
        return None

    results = {}

    for result_file in results_dir.rglob("*.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'resolved' in data:
                results.update(data)

    if not results:
        for result_file in results_dir.rglob("*.jsonl"):
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        instance_id = item.get('instance_id')
                        if instance_id:
                            results[instance_id] = item

    return results


async def generate_predictions(args, dataset):
    logger.info(f"Starting prediction generation with model: {args.llm_name}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == 'direct':
        agent_names = ['CodeWriting']
        kwargs = get_kwargs('direct', 1)
        num_rounds = 1
    else:
        agent_names = [name for name in args.agent_names for _ in range(args.agent_nums)]
        kwargs = get_kwargs('fullconnect', len(agent_names))
        num_rounds = args.num_rounds

    logger.info(f"Number of agents: {len(agent_names)}")
    logger.info(f"Number of rounds: {num_rounds}")

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=False,
        optimized_temporal=False,
        rounds=num_rounds,
        **kwargs
    )

    max_samples = args.max_samples or len(dataset)
    predictions = []
    errors = []

    for i, record in enumerate(dataset):
        if i >= max_samples:
            break

        instance_id = record.get('instance_id', str(i))
        repo = record.get('repo', '')

        logger.info(f"Processing sample {i + 1}/{max_samples} - Instance ID: {instance_id}")
        logger.debug(f"Repository: {repo}")

        try:
            input_dict = SWEBenchLiteDataset.record_to_input(record)
            answer = await graph.arun(input_dict, num_rounds=num_rounds)

            if isinstance(answer, list):
                answer_text = answer[0] if answer else ""
            else:
                answer_text = str(answer)

            model_patch = SWEBenchLiteDataset.postprocess_answer(answer_text)

            prediction_item = {
                "instance_id": instance_id,
                "model_name_or_path": args.llm_name,
                "model_patch": model_patch
            }
            predictions.append(prediction_item)

            logger.info(f"Sample {i + 1}: Generated patch (length: {len(model_patch)} chars)")

        except Exception as e:
            logger.error(f"Error processing sample {i + 1} (Instance ID: {instance_id}): {str(e)}")
            errors.append({
                "instance_id": instance_id,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

    return predictions, errors


async def main():
    args = parse_args()
    configure_thinking(
        enable_thinking=args.llm_enable_thinking,
        thinking_budget=args.llm_thinking_budget,
    )

    if args.debug:
        logger.setLevel(logging.DEBUG)

    cmd_line = " ".join([sys.executable] + sys.argv)
    logger.info(f"=" * 80)
    logger.info(f"SWE-bench Lite Evaluation Script (Official Mode)")
    logger.info(f"=" * 80)
    logger.info(f"Command: {cmd_line}")
    logger.info(f"=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"LLM Model: {args.llm_name}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Split: {args.split}")
    if args.mode == 'fullconnect':
        logger.info(f"Number of Agents: {args.agent_nums}")
        logger.info(f"Number of Rounds: {args.num_rounds}")
    if args.max_samples:
        logger.info(f"Max Samples: {args.max_samples}")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"=" * 80)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = Path(f"{AgentPrune_ROOT}/result/swe_bench_lite/{args.run_id}")
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {result_dir}")

    predictions_path = result_dir / "predictions.jsonl"
    errors_path = result_dir / "errors.json"
    final_results_path = result_dir / "final_results.json"

    logger.info(f"Predictions will be saved to: {predictions_path}")

    start_time = time.time()

    try:
        logger.info("Loading SWE-bench Lite dataset...")
        dataset = SWEBenchLiteDataset(
            dataset_dir="dataset/swe_bench_lite",
            split=args.split,
            use_huggingface=True
        )
        logger.info(f"Successfully loaded SWE-bench Lite dataset")
        logger.info(f"Total samples: {len(dataset)}")

        categories = dataset.get_categories()
        logger.info(f"Categories (repos): {categories}")
        for cat in categories:
            count = len(dataset.get_records_by_category(cat))
            logger.info(f"  - {cat}: {count} samples")

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    logger.info(f"=" * 80)
    logger.info(f"Generating predictions...")
    logger.info(f"=" * 80)

    predictions, errors = await generate_predictions(args, dataset)

    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Encountered {len(errors)} errors")

    with open(predictions_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    logger.info(f"Predictions saved to: {predictions_path}")

    if errors:
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=4, ensure_ascii=False)
        logger.info(f"Errors saved to: {errors_path}")

    if args.skip_evaluation:
        logger.info("Skipping official evaluation as requested")
        end_time = time.time()

        output = {
            "command": cmd_line,
            "metadata": {
                "model": args.llm_name,
                "dataset": args.dataset_name,
                "mode": args.mode,
                "split": args.split,
                "agent_nums": args.agent_nums if args.mode == 'fullconnect' else 1,
                "num_rounds": args.num_rounds if args.mode == 'fullconnect' else 1,
                "max_samples": args.max_samples or len(dataset),
                "run_id": args.run_id,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                "duration_seconds": round(end_time - start_time, 2),
                "cost": Cost.instance().value,
                "prompt_tokens": PromptTokens.instance().value,
                "completion_tokens": CompletionTokens.instance().value,
                "num_predictions": len(predictions),
                "num_errors": len(errors)
            },
            "predictions_file": str(predictions_path),
            "errors_file": str(errors_path) if errors else None
        }

        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(f"=" * 80)
        logger.info(f"Prediction Generation Complete")
        logger.info(f"=" * 80)
        logger.info(f"Duration: {round(end_time - start_time, 2)} seconds")
        logger.info(f"Cost: {Cost.instance().value}")
        logger.info(f"Prompt Tokens: {PromptTokens.instance().value}")
        logger.info(f"Completion Tokens: {CompletionTokens.instance().value}")
        logger.info(f"Results saved to: {final_results_path}")
        return

    logger.info(f"=" * 80)
    logger.info(f"Running official evaluation...")
    logger.info(f"=" * 80)

    eval_results_dir = run_official_evaluation(
        predictions_path,
        args.dataset_name,
        args.run_id,
        args.max_workers
    )

    official_results = parse_official_results(eval_results_dir)

    end_time = time.time()

    if official_results:
        total_samples = len(predictions)
        resolved = sum(1 for v in official_results.values() if v.get('resolved', False))
        overall_accuracy = resolved / total_samples if total_samples > 0 else 0.0

        category_accuracy = {}
        for record in dataset:
            instance_id = record.get('instance_id')
            repo = record.get('repo', '')
            repo_name = repo.split("/")[-1] if "/" in repo else repo

            if repo_name not in category_accuracy:
                category_accuracy[repo_name] = {'total': 0, 'resolved': 0}

            category_accuracy[repo_name]['total'] += 1
            if official_results.get(instance_id, {}).get('resolved', False):
                category_accuracy[repo_name]['resolved'] += 1

        for repo_name, stats in category_accuracy.items():
            stats['accuracy'] = stats['resolved'] / stats['total'] if stats['total'] > 0 else 0.0
    else:
        total_samples = len(predictions)
        resolved = 0
        overall_accuracy = 0.0
        category_accuracy = {}

    output = {
        "command": cmd_line,
        "metadata": {
            "model": args.llm_name,
            "dataset": args.dataset_name,
            "mode": args.mode,
            "split": args.split,
            "agent_nums": args.agent_nums if args.mode == 'fullconnect' else 1,
            "num_rounds": args.num_rounds if args.mode == 'fullconnect' else 1,
            "max_samples": args.max_samples or len(dataset),
            "run_id": args.run_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "duration_seconds": round(end_time - start_time, 2),
            "cost": Cost.instance().value,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value
        },
        "metrics": {
            "total_samples": total_samples,
            "total_resolved": resolved,
            "overall_accuracy": overall_accuracy,
            "category_accuracy": category_accuracy
        },
        "predictions_file": str(predictions_path),
        "errors_file": str(errors_path) if errors else None,
        "official_results": official_results if official_results else {}
    }

    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    logger.info(f"=" * 80)
    logger.info(f"Evaluation Complete")
    logger.info(f"=" * 80)
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Total Resolved: {resolved}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Duration: {round(end_time - start_time, 2)} seconds")
    logger.info(f"Cost: {Cost.instance().value}")
    logger.info(f"Prompt Tokens: {PromptTokens.instance().value}")
    logger.info(f"Completion Tokens: {CompletionTokens.instance().value}")

    if category_accuracy:
        logger.info(f"\nCategory-wise Accuracy:")
        for repo_name, stats in category_accuracy.items():
            logger.info(f"  {repo_name}: {stats['resolved']}/{stats['total']} ({stats['accuracy']:.4f})")

    logger.info(f"\nResults saved to: {final_results_path}")

    if eval_results_dir:
        logger.info(f"Official evaluation logs saved to: {eval_results_dir}")


if __name__ == '__main__':
    asyncio.run(main())