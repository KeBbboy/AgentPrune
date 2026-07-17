"""消息压缩框架的配置层.

3 个开关 ↔ 3 个独立模块(每个开关对应一个模块):
  - ``role_aware_compression``  -> :mod:`.role_aware_compression`
  - ``node_aware_aggregation``  -> :mod:`.node_aware_aggregation`
  - ``temporal_api_summary``    -> :mod:`.temporal_api_summary`

设计要点:
  - 仅 3 个 bool 字段, 默认全 False, 与基线行为一致
  - argparse 只注册这 3 个开关, 不再含历史 ``temporal_summary_mode`` /
    ``temporal_full_history_edges`` 等已下沉的开关
  - 与 ``compress_messages`` 的全局单例风格保持一致
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MessageCompressionConfig:
    """消息压缩框架的运行期配置 (3 个独立开关)."""

    role_aware_compression: bool = False
    node_aware_aggregation: bool = False
    temporal_api_summary: bool = False

    # 内部开关, 不在 argparse 上注册, 也不写入 ``extra``. 仅供单元测试使用:
    # 当 ``True`` 时, 模块 1/2 在 ``message_type != "structured"`` 时仍抛
    # :class:`UnstructuredPayloadError` 验证设计意图; 运行时默认 ``False``
    # (生产中 Node.async_execute 在压扁 LLMResponse 后 message_type 永远是
    # ``"full"``, 严格报错会让所有 node 100% 抛错, Graph 没法产生候选答案).
    strict_unstructured: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        # 三个开关都是 bool, 无需范围校验; 留作扩展钩子
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
    """在 runner 的 argparse 上注册 3 个开关, 每个对应一个独立模块."""
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


def build_message_compression_config_from_args(args) -> MessageCompressionConfig:
    cfg = MessageCompressionConfig(
        role_aware_compression=args.role_aware_compression,
        node_aware_aggregation=args.node_aware_aggregation,
        temporal_api_summary=args.temporal_api_summary,
    )
    cfg.validate()
    return cfg
