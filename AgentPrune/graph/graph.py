import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio
import tiktoken

from AgentPrune.graph.node import Node
from AgentPrune.agents.agent_registry import AgentRegistry

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                log_node_outputs: bool = False,
                llm_temperature: float = 0.0,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                edge_message_mode: str = "full",
                edge_answer_percent: float = 0.0,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"

        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.llm_temperature: float = llm_temperature
        self.agent_names:List[str] = agent_names
        self.log_node_outputs: bool = log_node_outputs
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.edge_message_mode = edge_message_mode
        self.edge_answer_percent = edge_answer_percent
        self.eval_mode = False
        self.edge_content_policy = None
        self.edge_policy_training = False
        self.edge_policy_min_spatial_step_edges = 0
        self.edge_policy_min_temporal_step_edges = 0
        self.edge_policy_log_probs: List[torch.Tensor] = []
        self.edge_policy_entropies: List[torch.Tensor] = []
        self.edge_policy_action_records: List[Dict[str, Any]] = []
        self.edge_policy_action_counts: Dict[str, int] = {
            "answer": 0,
            "reasoning": 0,
            "spatial_answer": 0,
            "spatial_reasoning": 0,
            "temporal_answer": 0,
            "temporal_reasoning": 0,
        }
        self.current_run_prompt_tokens = 0
        self.current_run_completion_tokens = 0
        self.current_run_total_tokens = 0
        self.current_run_step_tokens = 0
        self.current_run_step_edges = 0
        self.current_run_spatial_step_tokens = 0
        self.current_run_temporal_step_tokens = 0
        self.current_run_step_tokens_by_edge: Dict[tuple[str, int], int] = {}
        self.enable_edge_cost_recording = True
        self.current_sample_index: Optional[int] = None
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.spatial_edge_index_map: Dict[tuple[str, str], int] = {}
        self.temporal_edge_index_map: Dict[tuple[str, str], int] = {}
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self._token_encoder = None
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.spatial_edge_token_totals: Dict[tuple[str, str], int] = {
            (src_id, dst_id): 0
            for src_id, dst_id in self.potential_spatial_edges
            if src_id != dst_id
        }
        self.spatial_edge_usage_counts: Dict[tuple[str, str], int] = {
            (src_id, dst_id): 0
            for src_id, dst_id in self.potential_spatial_edges
            if src_id != dst_id
        }
        self.agent_token_stats: Dict[str, Dict[str, Any]] = {}
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
                                                 requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks
        self.initial_spatial_masks = fixed_spatial_masks.clone()

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
        self.initial_temporal_masks = fixed_temporal_masks.clone()

    def _log_node_output(self, node: Node, phase: str = "round") -> None:
        if not self.log_node_outputs:
            return

        raw_output = self._get_latest_output(node)
        extracted_answer = None
        if hasattr(node, "prompt_set") and hasattr(node.prompt_set, "postprocess_answer"):
            try:
                extracted_answer = node.prompt_set.postprocess_answer(raw_output)
            except Exception:
                extracted_answer = None

        display_name = node.role or node.agent_name or node.id
        sample_tag = (
            f" sample_index={self.current_sample_index}"
            if self.current_sample_index is not None
            else ""
        )
        print(
            f"[NodeOutput][{phase}]{sample_tag} agent={repr(display_name)} id={node.id} role={repr(node.role)} "
            f"extracted_answer={repr(extracted_answer)} raw_output={repr(raw_output)}"
        )
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])
        self.spatial_edge_index_map = {
            (src_id, dst_id): edge_idx
            for edge_idx, (src_id, dst_id) in enumerate(self.potential_spatial_edges)
        }
        self.temporal_edge_index_map = {
            (src_id, dst_id): edge_idx
            for edge_idx, (src_id, dst_id) in enumerate(self.potential_temporal_edges)
        }

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
            self.nodes[node_id].spatial_message_types = {}
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
        self.decision_node.spatial_message_types = {}

    def reset_edge_policy_trace(self):
        self.edge_policy_log_probs = []
        self.edge_policy_entropies = []
        self.edge_policy_action_records = []
        self.edge_policy_action_counts = {
            "answer": 0,
            "reasoning": 0,
            "spatial_answer": 0,
            "spatial_reasoning": 0,
            "temporal_answer": 0,
            "temporal_reasoning": 0,
        }

    def reset_run_token_usage(self):
        self.current_run_prompt_tokens = 0
        self.current_run_completion_tokens = 0
        self.current_run_total_tokens = 0
        self.current_run_step_tokens = 0
        self.current_run_step_edges = 0
        self.current_run_spatial_step_tokens = 0
        self.current_run_temporal_step_tokens = 0
        self.current_run_step_tokens_by_edge = {}

    def get_edge_policy_log_prob_sum(self) -> torch.Tensor:
        if len(self.edge_policy_log_probs) == 0:
            return torch.tensor(0.0)
        return torch.sum(torch.stack(self.edge_policy_log_probs))

    def get_edge_policy_entropy_sum(self) -> torch.Tensor:
        if len(self.edge_policy_entropies) == 0:
            return torch.tensor(0.0)
        return torch.sum(torch.stack(self.edge_policy_entropies))

    def get_edge_policy_action_counts(self) -> Dict[str, int]:
        return dict(self.edge_policy_action_counts)

    def get_edge_policy_action_records(self) -> List[Dict[str, Any]]:
        return list(self.edge_policy_action_records)

    def get_step_tokens_by_edge(self) -> Dict[tuple[str, int], int]:
        return dict(self.current_run_step_tokens_by_edge)

    def _get_eval_min_step_edges(self, edge_kind: str) -> int:
        if edge_kind == "spatial":
            return max(0, int(self.edge_policy_min_spatial_step_edges))
        return max(0, int(self.edge_policy_min_temporal_step_edges))

    def _get_active_edge_indices(self, edge_kind: str) -> List[int]:
        if edge_kind == "spatial":
            return [
                edge_idx
                for edge_idx, ((src_id, dst_id), edge_mask) in enumerate(
                    zip(self.potential_spatial_edges, self.spatial_masks)
                )
                if src_id != dst_id and edge_mask > 0
            ]
        return [
            edge_idx
            for edge_idx, edge_mask in enumerate(self.temporal_masks)
            if edge_mask > 0
        ]

    def _get_eval_reasoning_edge_indices(self, edge_kind: str) -> set[int]:
        if self.edge_content_policy is None:
            return set()

        policy = self.edge_content_policy
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = self.spatial_logits.device

        active_edge_indices = self._get_active_edge_indices(edge_kind)
        if not active_edge_indices:
            return set()

        edge_step_scores: List[tuple[int, float]] = []
        reasoning_edge_indices: set[int] = set()

        for edge_idx in active_edge_indices:
            edge_feature = self.get_edge_policy_feature(edge_idx, edge_kind, device=device)
            with torch.no_grad():
                probs = torch.softmax(policy(edge_feature), dim=-1).squeeze(0)
            answer_prob = float(probs[0].item())
            step_prob = float(probs[1].item())
            edge_step_scores.append((edge_idx, step_prob))
            if step_prob > answer_prob:
                reasoning_edge_indices.add(edge_idx)

        min_step_edges = min(self._get_eval_min_step_edges(edge_kind), len(active_edge_indices))
        if len(reasoning_edge_indices) < min_step_edges:
            for edge_idx, _step_prob in sorted(edge_step_scores, key=lambda item: item[1], reverse=True):
                reasoning_edge_indices.add(edge_idx)
                if len(reasoning_edge_indices) >= min_step_edges:
                    break

        return reasoning_edge_indices

    def get_edge_policy_feature_dim(self) -> int:
        return 2 + 2 * len(self.nodes)

    def get_edge_policy_feature(
        self,
        edge_idx: int,
        edge_kind: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        edge_logits = self.spatial_logits if edge_kind == "spatial" else self.temporal_logits
        potential_edges = (
            self.potential_spatial_edges
            if edge_kind == "spatial"
            else self.potential_temporal_edges
        )
        src_id, dst_id = potential_edges[edge_idx]
        node_ids = list(self.nodes.keys())
        src_index = node_ids.index(src_id)
        dst_index = node_ids.index(dst_id)
        edge_weight = float(torch.sigmoid(edge_logits[edge_idx].detach()).item())
        feature = torch.zeros(self.get_edge_policy_feature_dim(), dtype=torch.float32)
        feature[0] = edge_weight
        feature[1] = 1.0 if edge_kind == "temporal" else 0.0
        feature[2 + src_index] = 1.0
        feature[2 + len(node_ids) + dst_index] = 1.0
        if device is not None:
            feature = feature.to(device)
        return feature.view(1, -1)

    def _normalized_edge_answer_ratio(self) -> float:
        ratio = self.edge_answer_percent
        if ratio > 1.0:
            ratio = ratio / 100.0
        return min(max(ratio, 0.0), 1.0)

    def _top_percent_answer_edge_indices(self) -> set[int]:
        active_edge_indices = [
            edge_idx
            for edge_idx, ((src_id, dst_id), edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_masks))
            if edge_mask > 0 and src_id != dst_id
        ]
        if len(active_edge_indices) == 0:
            return set()

        answer_ratio = self._normalized_edge_answer_ratio()
        answer_edge_count = int(np.ceil(len(active_edge_indices) * answer_ratio))
        if answer_edge_count <= 0:
            return set()

        edge_weights = torch.sigmoid(self.spatial_logits.detach())
        sorted_edge_indices = sorted(
            active_edge_indices,
            key=lambda edge_idx: float(edge_weights[edge_idx]),
            reverse=True,
        )
        return set(sorted_edge_indices[:answer_edge_count])

    def _top_percent_answer_temporal_edge_indices(self) -> set[int]:
        active_edge_indices = [
            edge_idx
            for edge_idx, (edge_mask) in enumerate(self.temporal_masks)
            if edge_mask > 0
        ]
        if len(active_edge_indices) == 0:
            return set()

        answer_ratio = self._normalized_edge_answer_ratio()
        answer_edge_count = int(np.ceil(len(active_edge_indices) * answer_ratio))
        if answer_edge_count <= 0:
            return set()

        edge_weights = torch.sigmoid(self.temporal_logits.detach())
        sorted_edge_indices = sorted(
            active_edge_indices,
            key=lambda edge_idx: float(edge_weights[edge_idx]),
            reverse=True,
        )
        return set(sorted_edge_indices[:answer_edge_count])

    def _record_edge_policy_action(self, edge_kind: str, message_type: str) -> None:
        self.edge_policy_action_counts[message_type] += 1
        self.edge_policy_action_counts[f"{edge_kind}_{message_type}"] += 1

    def _get_rl_policy_message_type(self, edge_idx: int, edge_kind: str) -> str:
        if self.edge_content_policy is None:
            return "full"

        policy = self.edge_content_policy
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = self.spatial_logits.device
        edge_feature = self.get_edge_policy_feature(edge_idx, edge_kind, device=device)
        logits = policy(edge_feature)

        if self.edge_policy_training:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()
            entropy = dist.entropy().squeeze()
            self.edge_policy_log_probs.append(log_prob)
            self.edge_policy_entropies.append(entropy)
            action_value = int(action.item())
        else:
            log_prob = None
            entropy = None
            reasoning_edge_indices = self._get_eval_reasoning_edge_indices(edge_kind)
            action_value = 1 if edge_idx in reasoning_edge_indices else 0

        message_type = "answer" if action_value == 0 else "reasoning"
        self._record_edge_policy_action(edge_kind, message_type)
        if self.edge_policy_training and log_prob is not None and entropy is not None:
            self.edge_policy_action_records.append({
                "edge_kind": edge_kind,
                "edge_idx": edge_idx,
                "message_type": message_type,
                "log_prob": log_prob,
                "entropy": entropy,
            })
        return message_type

    def _get_edge_message_type(
        self,
        edge_idx: int,
        edge_kind: str,
        answer_edge_indices: set[int],
    ) -> str:
        if self.edge_message_mode == "full" or not self.eval_mode:
            return "full"
        if self.edge_message_mode == "top_percent":
            message_type = "answer" if edge_idx in answer_edge_indices else "reasoning"
            self._record_edge_policy_action(edge_kind, message_type)
            return message_type
        if self.edge_message_mode == "rl_policy":
            return self._get_rl_policy_message_type(edge_idx, edge_kind)
        raise ValueError(
            f"Unsupported edge_message_mode: {self.edge_message_mode}. "
            "Expected one of: full, top_percent, rl_policy."
        )

    def _add_spatial_successor_with_message_type(
        self,
        out_node: Node,
        in_node: Node,
        edge_idx: int,
        answer_edge_indices: set[int],
    ):
        out_node.add_successor(in_node, 'spatial')
        in_node.spatial_message_types[out_node.id] = self._get_edge_message_type(
            edge_idx,
            "spatial",
            answer_edge_indices,
        )

    def _add_temporal_successor_with_message_type(
        self,
        out_node: Node,
        in_node: Node,
        edge_idx: int,
        answer_edge_indices: set[int],
    ):
        out_node.add_successor(in_node, 'temporal')
        in_node.temporal_message_types[out_node.id] = self._get_edge_message_type(
            edge_idx,
            "temporal",
            answer_edge_indices,
        )

    def _get_token_encoder(self):
        if self._token_encoder is not None:
            return self._token_encoder
        try:
            self._token_encoder = tiktoken.encoding_for_model(self.llm_name or "gpt-4o")
        except Exception:
            self._token_encoder = tiktoken.get_encoding("cl100k_base")
        return self._token_encoder

    def count_text_tokens(self, text: Any) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        try:
            return len(self._get_token_encoder().encode(text))
        except Exception:
            return len(text.split())

    def count_message_tokens(self, messages: Any) -> int:
        if messages is None:
            return 0
        if not isinstance(messages, list):
            return self.count_text_tokens(messages)

        token_count = 0
        for message in messages:
            if isinstance(message, dict):
                token_count += self.count_text_tokens(message.get("role", ""))
                token_count += self.count_text_tokens(message.get("content", ""))
            else:
                token_count += self.count_text_tokens(message)
        return token_count

    def _get_latest_output(self, node: Node) -> Any:
        outputs = node.outputs
        if isinstance(outputs, list):
            if len(outputs) == 0:
                return None
            return outputs[-1]
        return outputs

    def _get_latest_memory_output(self, node: Node) -> Any:
        outputs = node.last_memory.get("outputs", [])
        if isinstance(outputs, list):
            if len(outputs) == 0:
                return None
            return outputs[-1]
        return outputs

    def record_step_edge_token_usage(self, node: Node):
        for predecessor in node.spatial_predecessors:
            message_type = node.spatial_message_types.get(predecessor.id, "full")
            if message_type != "reasoning":
                continue
            formatted_output = node._format_edge_output(
                predecessor=predecessor,
                output=self._get_latest_output(predecessor),
                message_type=message_type,
            )
            token_cost = self.count_text_tokens(formatted_output)
            self.current_run_step_tokens += token_cost
            self.current_run_spatial_step_tokens += token_cost
            self.current_run_step_edges += 1
            edge_idx = self.spatial_edge_index_map.get((predecessor.id, node.id))
            if edge_idx is not None:
                edge_key = ("spatial", edge_idx)
                self.current_run_step_tokens_by_edge[edge_key] = (
                    self.current_run_step_tokens_by_edge.get(edge_key, 0) + token_cost
                )

        for predecessor in node.temporal_predecessors:
            message_type = node.temporal_message_types.get(predecessor.id, "full")
            if message_type != "reasoning":
                continue
            formatted_output = node._format_edge_output(
                predecessor=predecessor,
                output=self._get_latest_memory_output(predecessor),
                message_type=message_type,
            )
            token_cost = self.count_text_tokens(formatted_output)
            self.current_run_step_tokens += token_cost
            self.current_run_temporal_step_tokens += token_cost
            self.current_run_step_edges += 1
            edge_idx = self.temporal_edge_index_map.get((predecessor.id, node.id))
            if edge_idx is not None:
                edge_key = ("temporal", edge_idx)
                self.current_run_step_tokens_by_edge[edge_key] = (
                    self.current_run_step_tokens_by_edge.get(edge_key, 0) + token_cost
                )

    def record_spatial_edge_costs(self, node: Node):
        if not self.enable_edge_cost_recording:
            return
        for predecessor in node.spatial_predecessors:
            if predecessor.id == node.id:
                continue
            edge_key = (predecessor.id, node.id)
            if edge_key not in self.spatial_edge_token_totals:
                continue
            token_cost = self.count_text_tokens(self._get_latest_output(predecessor))
            self.spatial_edge_token_totals[edge_key] += token_cost
            self.spatial_edge_usage_counts[edge_key] += 1

    def share_cost_state(self, other: "Graph"):
        self.spatial_edge_token_totals = other.spatial_edge_token_totals
        self.spatial_edge_usage_counts = other.spatial_edge_usage_counts
        self.agent_token_stats = other.agent_token_stats
        self._token_encoder = other._token_encoder
        self.enable_edge_cost_recording = other.enable_edge_cost_recording

    def record_node_token_usage(self, node: Node):
        prompt_tokens = None
        completion_tokens = None
        if node.last_llm_usage is not None:
            prompt_tokens = node.last_llm_usage.get("prompt_tokens")
            completion_tokens = node.last_llm_usage.get("completion_tokens")

        if prompt_tokens is None:
            prompt_tokens = self.count_message_tokens(node.last_input_messages)
        if completion_tokens is None:
            completion_tokens = self.count_text_tokens(self._get_latest_output(node))
        self.current_run_prompt_tokens += prompt_tokens
        self.current_run_completion_tokens += completion_tokens
        self.current_run_total_tokens += prompt_tokens + completion_tokens
        if node is self.decision_node:
            return
        node_key = node.id

        if node_key not in self.agent_token_stats:
            self.agent_token_stats[node_key] = {
                "id": node.id,
                "agent_name": node.agent_name,
                "role": node.role or node.node_name,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }

        stats = self.agent_token_stats[node_key]
        stats["agent_name"] = node.agent_name
        stats["role"] = node.role or node.node_name
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["total_tokens"] += prompt_tokens + completion_tokens
        stats["calls"] += 1

    def get_agent_token_metrics(self):
        return sorted(
            self.agent_token_stats.values(),
            key=lambda item: (-item["total_tokens"], item["role"], item["id"]),
        )

    def get_role_token_metrics(self):
        role_metrics: Dict[str, Dict[str, Any]] = {}
        for item in self.agent_token_stats.values():
            role = item["role"]
            if role not in role_metrics:
                role_metrics[role] = {
                    "role": role,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                    "agents": 0,
                }
            role_metrics[role]["prompt_tokens"] += item["prompt_tokens"]
            role_metrics[role]["completion_tokens"] += item["completion_tokens"]
            role_metrics[role]["total_tokens"] += item["total_tokens"]
            role_metrics[role]["calls"] += item["calls"]
            role_metrics[role]["agents"] += 1
        return sorted(
            role_metrics.values(),
            key=lambda item: (-item["total_tokens"], item["role"]),
        )
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []
            self.nodes[node_id].temporal_message_types = {}

    def reset_runtime_audit(self):
        for node in self.nodes.values():
            node.reset_runtime_audit()
        self.decision_node.reset_runtime_audit()

    def connect_decision_node(self, last_node_id: str = None):
        for node_id in self.nodes.keys():
            if last_node_id is None:
                self.nodes[node_id].add_successor(self.decision_node)
            elif last_node_id == node_id:
                self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        answer_edge_indices = self._top_percent_answer_edge_indices()
        
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    self._add_spatial_successor_with_message_type(
                        out_node,
                        in_node,
                        edge_idx,
                        answer_edge_indices,
                    )
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    self._add_spatial_successor_with_message_type(
                        out_node,
                        in_node,
                        edge_idx,
                        answer_edge_indices,
                    )
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        answer_edge_indices = self._top_percent_answer_temporal_edge_indices()
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    self._add_temporal_successor_with_message_type(
                        out_node,
                        in_node,
                        edge_idx,
                        answer_edge_indices,
                    )
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                self._add_temporal_successor_with_message_type(
                    out_node,
                    in_node,
                    edge_idx,
                    answer_edge_indices,
                )
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  aggregate_mode: str = "all connected") -> List[Any]:
        # inputs:{'task':"xxx"}
        self.reset_runtime_audit()
        self.reset_edge_policy_trace()
        self.reset_run_token_usage()
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            execution_index = 0

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                current_node = self.nodes[current_node_id]
                current_node.runtime_sample_index = self.current_sample_index
                current_node.runtime_round_index = round
                current_node.runtime_execution_index = execution_index
                tries = 0
                while tries < max_tries:
                    try:
                        self.record_spatial_edge_costs(current_node)
                        self.record_step_edge_token_usage(current_node)
                        current_node.execute(inputs, temperature=self.llm_temperature) # output is saved in the node.outputs
                        self.record_node_token_usage(current_node)
                        self._log_node_output(current_node, phase=f"round-{round}")
                        break
                    except Exception as e:
                        display_name = current_node.role or current_node.agent_name or current_node.id
                        print(f"Error during execution of agent {repr(display_name)} (id={current_node_id}): {e}")
                    tries += 1
                execution_index += 1
                for successor in current_node.spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        self.decision_node.execute(inputs, temperature=self.llm_temperature)
        self.record_node_token_usage(self.decision_node)
        self._log_node_output(self.decision_node, phase="final-decision")
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,
                  aggregate_mode: str = "all connected",) -> List[Any]:
        self.reset_runtime_audit()
        self.reset_edge_policy_trace()
        self.reset_run_token_usage()
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            execution_index = 0

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                current_node = self.nodes[current_node_id]
                current_node.runtime_sample_index = self.current_sample_index
                current_node.runtime_round_index = round
                current_node.runtime_execution_index = execution_index
                tries = 0
                while tries < max_tries:
                    try:
                        self.record_spatial_edge_costs(current_node)
                        self.record_step_edge_token_usage(current_node)
                        await asyncio.wait_for(
                            current_node.async_execute(input, temperature=self.llm_temperature),
                        timeout=max_time,
                        ) # output is saved in the node.outputs
                        self.record_node_token_usage(current_node)
                        self._log_node_output(current_node, phase=f"round-{round}")
                        break
                    except Exception as e:
                        display_name = current_node.role or current_node.agent_name or current_node.id
                        print(f"Error during execution of agent {repr(display_name)} (id={current_node_id}): {e}")
                    tries += 1
                execution_index += 1
                for successor in current_node.spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        if aggregate_mode == "all connected":
            self.connect_decision_node()
        elif aggregate_mode == "last connected":
            self.connect_decision_node(last_node_id=current_node_id)
        await self.decision_node.async_execute(input, temperature=self.llm_temperature)
        self.record_node_token_usage(self.decision_node)
        self._log_node_output(self.decision_node, phase="final-decision")
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()

    def set_eval_mode(self, enabled: bool):
        self.eval_mode = enabled
        self.enable_edge_cost_recording = not enabled

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
