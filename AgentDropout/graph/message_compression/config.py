"""消息压缩框架的配置层.

消息压缩框架配置层:
  - ``role_aware_compression``  -> :mod:`.role_aware_compression`
  - ``node_aware_aggregation``  -> :mod:`.node_aware_aggregation`
  - ``temporal_api_summary``    -> :mod:`.temporal_api_summary`
  - ``selective_context``       -> :mod:`AgentDropout.context.selective_context`
  - ``llmlingua2_compression``  -> :mod:`AgentDropout.utils.llmlingua2_compressor`

设计要点:
  - 所有压缩算法默认关闭, 与基线行为一致
  - argparse 集中注册现有压缩模块与新移植的两个算法开关
  - 与 ``compress_messages`` 的全局单例风格保持一致
"""
from dataclasses import dataclass, field
from typing import Dict, Any

from AgentDropout.utils.llmlingua2_compressor import DEFAULT_LLMLINGUA2_MODEL


@dataclass
class MessageCompressionConfig:
    """消息压缩框架的运行期配置."""

    role_aware_compression: bool = False
    node_aware_aggregation: bool = False
    temporal_api_summary: bool = False
    selective_context: bool = False
    llmlingua2_compression: bool = False
    llmlingua2_rate: float = 0.5
    llmlingua2_model_name: str = DEFAULT_LLMLINGUA2_MODEL
    llmlingua2_min_tokens: int = 20
    llmlingua2_compress_code: bool = False

    # 内部开关, 不在 argparse 上注册, 也不写入 ``extra``. 仅供单元测试使用:
    # 当 ``True`` 时, 模块 1/2 在 ``message_type != "structured"`` 时仍抛
    # :class:`UnstructuredPayloadError` 验证设计意图; 运行时默认 ``False``
    # (生产中 Node.async_execute 在压扁 LLMResponse 后 message_type 永远是
    # ``"full"``, 严格报错会让所有 node 100% 抛错, Graph 没法产生候选答案).
    strict_unstructured: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0 < float(self.llmlingua2_rate) <= 1:
            raise ValueError("llmlingua2_rate must be in (0, 1].")
        return None


_CONFIG = MessageCompressionConfig()


def get_message_compression_config() -> MessageCompressionConfig:
    return _CONFIG


def set_message_compression_config(cfg: MessageCompressionConfig) -> None:
    global _CONFIG
    cfg.validate()
    _CONFIG = cfg


def parse_bool_arg(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def add_message_compression_arguments(parser) -> None:
    """在 runner 的 argparse 上注册消息压缩相关开关."""
    parser.add_argument(
        "--role_aware_compression", type=parse_bool_arg, default=False,
        help="启用 role_aware_compression 模块: 按 self_role 截断前驱 reasoning 字段",
    )
    parser.add_argument(
        "--node_aware_aggregation", type=parse_bool_arg, default=False,
        help="启用 node_aware_aggregation 模块: 按 self_role 重新聚合前驱 (reasoning/answer 分别拼接)",
    )
    parser.add_argument(
        "--temporal_api_summary", type=parse_bool_arg, default=False,
        help="启用 temporal_api_summary 模块: Graph.arun 每轮末尾按 answer 一致性聚合多智能体输出",
    )
    parser.add_argument(
        "--selective_context",
        nargs="?",
        const=True,
        type=parse_bool_arg,
        default=False,
        help="启用 Selective Context: 按 self-information 删除低信息量消息片段",
    )
    parser.add_argument(
        "--llmlingua2_compression",
        nargs="?",
        const=True,
        type=parse_bool_arg,
        default=False,
        help="启用 LLMLingua-2 压缩智能体间消息",
    )
    parser.add_argument(
        "--llmlingua2_rate",
        type=float,
        default=0.5,
        help="LLMLingua-2 消息压缩保留率",
    )
    parser.add_argument(
        "--llmlingua2_model_name",
        type=str,
        default=DEFAULT_LLMLINGUA2_MODEL,
        help="LLMLingua-2 使用的 HuggingFace 模型",
    )
    parser.add_argument(
        "--llmlingua2_min_tokens",
        type=int,
        default=20,
        help="短于该近似 token 数的消息不做 LLMLingua-2 压缩",
    )
    parser.add_argument(
        "--llmlingua2_compress_code",
        nargs="?",
        const=True,
        type=parse_bool_arg,
        default=False,
        help="允许 LLMLingua-2 压缩 HumanEval 代码 payload",
    )

def build_message_compression_config_from_args(args) -> MessageCompressionConfig:
    cfg = MessageCompressionConfig(
        role_aware_compression=args.role_aware_compression,
        node_aware_aggregation=args.node_aware_aggregation,
        temporal_api_summary=args.temporal_api_summary,
        selective_context=getattr(args, "selective_context", False),
        llmlingua2_compression=getattr(args, "llmlingua2_compression", False),
        llmlingua2_rate=getattr(args, "llmlingua2_rate", 0.5),
        llmlingua2_model_name=getattr(args, "llmlingua2_model_name", DEFAULT_LLMLINGUA2_MODEL),
        llmlingua2_min_tokens=getattr(args, "llmlingua2_min_tokens", 20),
        llmlingua2_compress_code=getattr(args, "llmlingua2_compress_code", False),
    )
    cfg.validate()
    return cfg
