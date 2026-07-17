"""消息压缩框架 (message compression framework).

3 个开关 ↔ 3 个独立模块(每个开关对应一个模块):

- :mod:`.role_aware_compression`  ← 开关 ``--role_aware_compression``
  :func:`compress_messages_role_aware` (供 Node.get_*_info 调用)
- :mod:`.node_aware_aggregation`  ← 开关 ``--node_aware_aggregation``
  :func:`aggregate_node_aware` (供 Node.get_*_info 调用)
- :mod:`.temporal_api_summary`    ← 开关 ``--temporal_api_summary``
  :func:`apply_temporal_api_summary` (供 Graph.arun 每轮末尾调用)

对外统一入口:

- :class:`MessageCompressionConfig`  —— 3 字段配置
- :func:`get_message_compression_config` / :func:`set_message_compression_config`
- :func:`add_message_compression_arguments`  —— argparse 集成,只注册 3 个开关
- :func:`build_message_compression_config_from_args`
"""
from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
    add_message_compression_arguments,
    build_message_compression_config_from_args,
    get_message_compression_config,
    parse_bool_arg,
    set_message_compression_config,
)
from AgentDropout.graph.message_compression.role_aware_compression import (
    UnstructuredPayloadError as RoleAwareUnstructuredPayloadError,
    compress_messages_role_aware,
)
from AgentDropout.graph.message_compression.node_aware_aggregation import (
    UnstructuredPayloadError as AggregationUnstructuredPayloadError,
    aggregate_node_aware,
)
from AgentDropout.graph.message_compression.temporal_api_summary import (
    apply_temporal_api_summary,
)


__all__ = [
    # --- 配置层 ---
    "MessageCompressionConfig",
    "add_message_compression_arguments",
    "build_message_compression_config_from_args",
    "get_message_compression_config",
    "parse_bool_arg",
    "set_message_compression_config",
    # --- 模块 1: role_aware_compression ---
    "RoleAwareUnstructuredPayloadError",
    "compress_messages_role_aware",
    # --- 模块 2: node_aware_aggregation ---
    "AggregationUnstructuredPayloadError",
    "aggregate_node_aware",
    # --- 模块 3: temporal_api_summary ---
    "apply_temporal_api_summary",
]
