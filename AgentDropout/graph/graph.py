import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import torch.nn.functional as F
import asyncio
import re
import hashlib

import AgentDropout.agents  # noqa: F401
import AgentDropout.prompt  # noqa: F401
from AgentDropout.graph.node import Node
from AgentDropout.agents.agent_registry import AgentRegistry
import random


MESSAGE_SELECTION_POLICIES = {"none", "binary", "aims", "aims_plus"}
MESSAGE_EMBEDDING_MODEL = None
MESSAGE_EMBEDDING_MODEL_FAILED = False


class MessageGraphAttentionSelector(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int = 4,
            ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
        self.gat_projection = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = torch.nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = torch.nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.similarity_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.output_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.attn_src, gain=0.1)
        torch.nn.init.xavier_uniform_(self.attn_dst, gain=0.1)

    def forward(self, message_embeddings: torch.Tensor) -> torch.Tensor:
        if message_embeddings.dim() == 1:
            message_embeddings = message_embeddings.unsqueeze(0)

        node_features = F.relu(self.input_projection(message_embeddings))
        projected = self.gat_projection(node_features).view(
            message_embeddings.size(0),
            self.num_heads,
            self.head_dim,
        )

        src_scores = (projected * self.attn_src).sum(dim=-1)
        dst_scores = (projected * self.attn_dst).sum(dim=-1)
        attention_scores = src_scores[:, None, :] + dst_scores[None, :, :]

        normalized_embeddings = F.normalize(message_embeddings, p=2, dim=-1)
        similarity = normalized_embeddings @ normalized_embeddings.transpose(0, 1)
        attention_scores = attention_scores + self.similarity_scale * similarity.unsqueeze(-1)
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)

        attention_weights = torch.softmax(attention_scores, dim=1)
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * projected.unsqueeze(0), dim=1)
        aggregated = aggregated.reshape(message_embeddings.size(0), self.hidden_dim)
        relation_features = self.layer_norm(
            node_features + F.relu(self.output_projection(aggregated))
        )
        return self.action_head(torch.cat([node_features, relation_features], dim=-1))


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
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                diff:bool = False,
                dec:bool = False,
                rounds: int = 1,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                message_selection_policy: str = "none",
                message_selection_temperature: float = 1.0,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks)
        self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks)
        fixed_spatial_masks = self.fixed_spatial_masks.view(-1)
        fixed_temporal_masks = self.fixed_temporal_masks.view(-1)
        # print(fixed_temporal_masks)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.optimized_spatial_message_type = False
        self.optimized_temporal_message_type = False
        self.enable_spatial_message_type_action = False
        self.enable_temporal_message_type_action = False
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.diff=diff
        self.rounds=rounds
        self.message_selection_policy = self._normalize_message_selection_policy(message_selection_policy)
        self.message_selection_temperature = max(float(message_selection_temperature), 1e-6)
        self.message_actions = self._message_actions_for_policy(self.message_selection_policy)
        self.message_action_costs = torch.tensor(
            [self._message_action_cost(action) for action in self.message_actions],
            dtype=torch.float32,
        )
        self.message_action_counts = self._empty_message_action_counts()
        self.message_action_prob_history = []
        self.message_embedding_dim = 384
        self.message_gat_hidden_dim = 128
        self.message_gat_heads = 4
        self.message_embedding_model = None
        self.message_embedding_model_failed = False
        self.message_embedding_cache = {}
        self.message_action_gat = MessageGraphAttentionSelector(
            input_dim=self.message_embedding_dim,
            hidden_dim=self.message_gat_hidden_dim,
            num_heads=self.message_gat_heads,
            output_dim=len(self.message_actions),
        )
        self.message_action_gat.requires_grad_(False)
        self.message_complexity_action_bias = torch.nn.Parameter(
            torch.tensor(
                [self._message_complexity_bias(action) for action in self.message_actions],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        # self.dec=False
        self.dec_1=False
        self.skip_nodes = []
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.spatial_edge_index = self._build_edge_index(self.potential_spatial_edges)
        self.temporal_edge_index = self._build_edge_index(self.potential_temporal_edges)
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        init_spatial_message_type_logits = self._initial_message_type_logits("spatial")
        init_temporal_message_type_logits = self._initial_message_type_logits("temporal")

        print(self.potential_spatial_edges)
        if dec:
            # self.decision_masks = torch.nn.Parameter(torch.ones(5),requires_grad=False)
            self.decision_logits = torch.ones(5) * torch.log(torch.tensor(1.0))
            # self.spatial_logits_1 = None
            # self.temporal_logits_1 = None
            # print(self.decision_mask)
        if not diff:
            self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks
            self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial) # trainable edge logits
            self.spatial_message_type_logits = torch.nn.Parameter(
                init_spatial_message_type_logits.clone(),
                requires_grad=False,
            )
            self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal) # trainable edge logits
            self.temporal_message_type_logits = torch.nn.Parameter(
                init_temporal_message_type_logits.clone(),
                requires_grad=False,
            )
            self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
            if dec:
                self.spatial_logits_1 = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial)
                self.temporal_logits_1 = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal)
            # print(self.spatial_logits)
            # print(self.spatial_masks)
        else:
            if dec:
                self.spatial_logits_1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial) for _ in range(rounds)])
                self.temporal_logits_1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal) for _ in range(rounds-1)])
            self.spatial_masks = torch.nn.ParameterList([torch.nn.Parameter(fixed_spatial_masks.clone(), requires_grad=False) for _ in range(rounds)])  # fixed edge masks
            self.spatial_logits = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial) for _ in range(rounds)])
            self.spatial_message_type_logits = torch.nn.ParameterList([
                torch.nn.Parameter(init_spatial_message_type_logits.clone(), requires_grad=False)
                for _ in range(rounds)
            ])
            self.temporal_logits = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal) for _ in range(rounds-1)])
            self.temporal_message_type_logits = torch.nn.ParameterList([
                torch.nn.Parameter(init_temporal_message_type_logits.clone(), requires_grad=False)
                for _ in range(rounds-1)
            ])
            self.temporal_masks = torch.nn.ParameterList([torch.nn.Parameter(fixed_temporal_masks.clone(), requires_grad=False) for _ in range(rounds-1)])

    def _normalize_message_selection_policy(self, policy: str) -> str:
        normalized_policy = (policy or "none").lower()
        if normalized_policy not in MESSAGE_SELECTION_POLICIES:
            raise ValueError(
                f"Unknown message_selection_policy: {policy}. "
                f"Expected one of {sorted(MESSAGE_SELECTION_POLICIES)}"
            )
        return normalized_policy

    def _message_actions_for_policy(self, policy: str):
        if policy == "aims_plus":
            return ("answer", "reason", "full")
        if policy == "aims":
            return ("full", "answer", "reason")
        return ("answer", "reason")

    def _message_action_cost(self, action: str) -> float:
        return {
            "answer": 0.25,
            "reason": 0.70,
            "full": 1.00,
        }[action]

    def _empty_message_action_counts(self):
        return {
            "total": {action: 0 for action in self.message_actions},
            "spatial": {action: 0 for action in self.message_actions},
            "temporal": {action: 0 for action in self.message_actions},
        }

    def reset_message_action_counts(self):
        self.message_action_counts = self._empty_message_action_counts()
        self.message_action_prob_history = []

    def _record_message_action(self, edge_kind: str, action: str):
        if action not in self.message_actions:
            return
        self.message_action_counts["total"][action] += 1
        if edge_kind in self.message_action_counts:
            self.message_action_counts[edge_kind][action] += 1

    def _record_message_action_probs(self, action_probs: torch.Tensor):
        self.message_action_prob_history.append(action_probs)

    def message_action_count_summary(self) -> str:
        def format_counts(scope: str) -> str:
            counts = self.message_action_counts.get(scope, {})
            return ", ".join(
                f"{action}={counts.get(action, 0)}"
                for action in self.message_actions
            )

        return (
            f"total({format_counts('total')}) "
            f"spatial({format_counts('spatial')}) "
            f"temporal({format_counts('temporal')})"
        )

    def _print_message_action_counts(self):
        if self._active_message_selection_policy() == "aims_plus":
            print(f"AIMS+ action counts: {self.message_action_count_summary()}")

    def _message_complexity_bias(self, action: str) -> float:
        return {
            "answer": -0.30,
            "reason": 0.20,
            "full": 0.35,
        }[action]

    def _role_prior_for_message_action(self, sender: Node, receiver: Node, action: str) -> float:
        sender_role = (sender.role or "").lower()
        receiver_role = (receiver.role or "").lower()
        prior = 0.0

        if action == "answer":
            prior += 0.35 if "math solver" in receiver_role else 0.0
            prior -= 0.25 if any(role in receiver_role for role in ("inspector", "analyst", "programming")) else 0.0
        elif action == "reason":
            prior += 0.35 if any(role in receiver_role for role in ("inspector", "analyst")) else 0.0
            prior += 0.20 if any(role in sender_role for role in ("analyst", "inspector")) else 0.0
        elif action == "full":
            prior += 0.40 if self.domain == "humaneval" else 0.0
            prior += 0.25 if "programming" in sender_role or "programming" in receiver_role else 0.0
            prior += 0.15 if any(role in receiver_role for role in ("inspector", "decision")) else 0.0
        return prior

    def _build_edge_index(self, edges):
        return {
            (sender_id, receiver_id): edge_idx
            for edge_idx, (sender_id, receiver_id) in enumerate(edges)
        }

    def _initial_message_type_logits(self, edge_kind: str) -> torch.Tensor:
        edges = self.potential_spatial_edges if edge_kind == "spatial" else self.potential_temporal_edges
        if self.message_selection_policy in {"aims", "aims_plus"}:
            priors = []
            for sender_id, receiver_id in edges:
                sender = self.find_node(sender_id)
                receiver = self.find_node(receiver_id)
                priors.append([
                    self._role_prior_for_message_action(sender, receiver, action)
                    for action in self.message_actions
                ])
            return torch.tensor(priors, dtype=torch.float32)
        return torch.zeros(len(edges), dtype=torch.float32)

    def _task_complexity(self, inputs: Any) -> float:
        if isinstance(inputs, dict):
            task_text = str(inputs.get("task", ""))
        else:
            task_text = str(inputs)
        if not task_text:
            return 0.0
        token_count = len(task_text.split())
        line_count = max(1, task_text.count("\n") + 1)
        return min(1.0, (token_count / 160.0) + (line_count / 80.0))

    def _active_message_selection_policy(self) -> str:
        if not (self.enable_spatial_message_type_action or self.enable_temporal_message_type_action):
            return "none"
        if self.message_selection_policy == "aims_plus":
            return "aims_plus"
        if self.message_selection_policy == "aims":
            return "aims"
        return "binary"
        
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
        for node in self.nodes.values():
            node.spatial_message_types = {}
            node.temporal_message_types = {}
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

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
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []
            self.nodes[node_id].temporal_message_types = {}

    def _iter_parameters(self, value):
        if value is None:
            return []
        if isinstance(value, torch.nn.Parameter):
            return [value]
        if isinstance(value, torch.nn.ParameterList):
            return list(value)
        if isinstance(value, torch.nn.Module):
            return list(value.parameters())
        return list(value)

    def _message_type_logits_for(self, edge_kind: str, round: int = 0):
        if edge_kind == "spatial":
            logits = self.spatial_message_type_logits
            return logits[round] if self.diff else logits
        if edge_kind == "temporal":
            logits = self.temporal_message_type_logits
            return logits[round - 1] if self.diff else logits
        raise ValueError(f"Unknown edge_kind: {edge_kind}")

    def sample_message_type(
            self,
            edge_kind: str,
            edge_idx: int,
            round: int = 0,
            return_log_prob: bool = False,
            task_complexity: float = 0.0,
            contextual_action_logits: Optional[torch.Tensor] = None,
            ):
        if edge_kind == "spatial":
            enabled = self.enable_spatial_message_type_action
            optimized = self.optimized_spatial_message_type
        elif edge_kind == "temporal":
            enabled = self.enable_temporal_message_type_action
            optimized = self.optimized_temporal_message_type
        else:
            raise ValueError(f"Unknown edge_kind: {edge_kind}")

        logits = self._message_type_logits_for(edge_kind, round)
        if not enabled:
            if return_log_prob:
                return "full", torch.tensor(0.0, device=logits.device)
            return "full"

        active_policy = self._active_message_selection_policy()
        if active_policy == "aims_plus" and contextual_action_logits is None:
            if return_log_prob:
                return "full", torch.tensor(0.0, device=logits.device)
            return "full"

        if active_policy in {"aims", "aims_plus"}:
            action_logits = logits[edge_idx]
            if active_policy == "aims_plus":
                action_logits = action_logits + contextual_action_logits.to(action_logits.device)
            action_logits = action_logits + self.message_complexity_action_bias.to(action_logits.device) * float(task_complexity)
            action_logits = action_logits / self.message_selection_temperature
            if not optimized:
                action_index = torch.argmax(action_logits)
                message_type = self.message_actions[int(action_index.item())]
                if return_log_prob:
                    return message_type, torch.tensor(0.0, device=action_logits.device)
                return message_type

            distribution = torch.distributions.Categorical(logits=action_logits)
            action_index = distribution.sample()
            message_type = self.message_actions[int(action_index.item())]
            if return_log_prob:
                return message_type, distribution.log_prob(action_index)
            return message_type

        if not optimized:
            message_type = "reason" if logits[edge_idx].item() >= 0 else "answer"
            if return_log_prob:
                return message_type, torch.tensor(0.0, device=logits.device)
            return message_type

        reason_prob = torch.sigmoid(logits[edge_idx])
        if torch.rand(1, device=reason_prob.device) < reason_prob:
            message_type = "reason"
            log_prob = torch.log(reason_prob)
        else:
            message_type = "answer"
            log_prob = torch.log(1 - reason_prob)
        if return_log_prob:
            return message_type, log_prob
        return message_type

    def contextual_message_regularization(
            self,
            entropy_weight: float = 0.0,
            balance_weight: float = 0.0,
            ) -> torch.Tensor:
        if self._active_message_selection_policy() != "aims_plus":
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)
        if not self.message_action_prob_history:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)
        if entropy_weight == 0.0 and balance_weight == 0.0:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)

        action_probs = torch.stack(self.message_action_prob_history)
        loss = torch.tensor(0.0, device=action_probs.device)
        if entropy_weight > 0.0:
            entropy = -(
                action_probs * torch.log(action_probs.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            loss = loss - entropy_weight * entropy
        if balance_weight > 0.0:
            target_values = {
                "answer": 0.30,
                "reason": 0.40,
                "full": 0.30,
            }
            target_distribution = torch.tensor(
                [
                    target_values.get(action, 1.0 / len(self.message_actions))
                    for action in self.message_actions
                ],
                dtype=action_probs.dtype,
                device=action_probs.device,
            )
            target_distribution = target_distribution / target_distribution.sum()
            average_probs = action_probs.mean(dim=0)
            balance_loss = torch.sum((average_probs - target_distribution) ** 2)
            loss = loss + balance_weight * balance_loss
        return loss

    def second_stage_optimizer_parameters(self):
        params = []
        params.extend(self._iter_parameters(self.spatial_logits))
        params.extend(self._iter_parameters(self.temporal_logits))
        if self.enable_spatial_message_type_action:
            params.extend(self._iter_parameters(self.spatial_message_type_logits))
        if self.enable_temporal_message_type_action:
            params.extend(self._iter_parameters(self.temporal_message_type_logits))
        if self._active_message_selection_policy() in {"aims", "aims_plus"}:
            params.extend(self._iter_parameters(self.message_complexity_action_bias))
        if self._active_message_selection_policy() == "aims_plus":
            params.extend(self._iter_parameters(self.message_action_gat))
        return params

    def share_second_stage_parameters(self, source: "Graph"):
        self.spatial_logits = source.spatial_logits
        self.temporal_logits = source.temporal_logits
        self.spatial_message_type_logits = source.spatial_message_type_logits
        self.temporal_message_type_logits = source.temporal_message_type_logits
        self.message_action_gat = source.message_action_gat
        self.message_complexity_action_bias = source.message_complexity_action_bias

    def set_message_type_learning_mode(self, enabled: bool):
        self.enable_spatial_message_type_action = enabled
        self.enable_temporal_message_type_action = enabled
        self.optimized_spatial_message_type = enabled and self.optimized_spatial
        self.optimized_temporal_message_type = enabled and self.optimized_temporal
        for param in self._iter_parameters(self.spatial_message_type_logits):
            param.requires_grad_(self.optimized_spatial_message_type)
        for param in self._iter_parameters(self.temporal_message_type_logits):
            param.requires_grad_(self.optimized_temporal_message_type)
        self.message_complexity_action_bias.requires_grad_(
            enabled
            and self._active_message_selection_policy() in {"aims", "aims_plus"}
            and (self.optimized_spatial or self.optimized_temporal)
        )
        self.message_action_gat.requires_grad_(
            enabled
            and self._active_message_selection_policy() == "aims_plus"
            and (self.optimized_spatial or self.optimized_temporal)
        )

    def message_selection_regularization(
            self,
            cost_weight: float = 0.0,
            entropy_weight: float = 0.0,
            diversity_weight: float = 0.0,
            context_weight_decay: float = 0.0,
            ) -> torch.Tensor:
        if self._active_message_selection_policy() not in {"aims", "aims_plus"}:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)
        if cost_weight == 0.0 and entropy_weight == 0.0 and diversity_weight == 0.0 and context_weight_decay == 0.0:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)

        cost_vector = self.message_action_costs.to(self.message_complexity_action_bias.device)
        cost_terms = []
        entropy_terms = []
        diversity_terms = []

        def collect(logits, masks, edges):
            active_mask = masks.to(logits.device).float()
            active_count = active_mask.sum().clamp_min(1.0)
            probs = torch.softmax(logits / self.message_selection_temperature, dim=-1)
            expected_cost = (probs * cost_vector.to(logits.device)).sum(dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            cost_terms.append((expected_cost * active_mask).sum() / active_count)
            entropy_terms.append((entropy * active_mask).sum() / active_count)

            receiver_ids = {receiver_id for _, receiver_id in edges}
            for receiver_id in receiver_ids:
                receiver_edge_indices = [
                    idx
                    for idx, (_, candidate_receiver_id) in enumerate(edges)
                    if candidate_receiver_id == receiver_id and float(active_mask[idx].detach().cpu()) > 0.0
                ]
                if len(receiver_edge_indices) < 2:
                    continue
                receiver_probs = probs[receiver_edge_indices].mean(dim=0)
                receiver_entropy = -(
                    receiver_probs * torch.log(receiver_probs.clamp_min(1e-8))
                ).sum()
                diversity_terms.append(receiver_entropy)

        if not self.diff:
            if self.enable_spatial_message_type_action:
                collect(self.spatial_message_type_logits, self.spatial_masks, self.potential_spatial_edges)
            if self.enable_temporal_message_type_action:
                collect(self.temporal_message_type_logits, self.temporal_masks, self.potential_temporal_edges)
        else:
            if self.enable_spatial_message_type_action:
                for logits, masks in zip(self.spatial_message_type_logits, self.spatial_masks):
                    collect(logits, masks, self.potential_spatial_edges)
            if self.enable_temporal_message_type_action:
                for logits, masks in zip(self.temporal_message_type_logits, self.temporal_masks):
                    collect(logits, masks, self.potential_temporal_edges)

        device = self.message_complexity_action_bias.device
        loss = torch.tensor(0.0, device=device)
        if cost_terms:
            loss = loss + cost_weight * torch.mean(torch.stack(cost_terms))
        if entropy_terms:
            loss = loss - entropy_weight * torch.mean(torch.stack(entropy_terms))
        if diversity_terms:
            loss = loss - diversity_weight * torch.mean(torch.stack(diversity_terms))
        if self._active_message_selection_policy() == "aims_plus":
            if context_weight_decay > 0.0:
                loss = loss + context_weight_decay * self._module_l2(self.message_action_gat)
        return loss

    def _module_l2(self, module: torch.nn.Module) -> torch.Tensor:
        params = [param.pow(2).mean() for param in module.parameters()]
        if not params:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)
        return torch.mean(torch.stack(params))

    def _latest_output(self, outputs):
        if isinstance(outputs, list) and len(outputs):
            return outputs[-1]
        if isinstance(outputs, list):
            return None
        return outputs

    def _incoming_message_candidates(self, receiver: Node):
        candidates = []
        for predecessor in receiver.spatial_predecessors:
            output = self._latest_output(predecessor.outputs)
            if output is not None:
                candidates.append(("spatial", predecessor, output))
        for predecessor in receiver.temporal_predecessors:
            output = self._latest_output(predecessor.last_memory["outputs"])
            if output is not None:
                candidates.append(("temporal", predecessor, output))
        return candidates

    def _sentence_embedding(self, text: Any) -> torch.Tensor:
        global MESSAGE_EMBEDDING_MODEL, MESSAGE_EMBEDDING_MODEL_FAILED
        text = str(text or "")
        cache_key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if cache_key in self.message_embedding_cache:
            return self.message_embedding_cache[cache_key].clone()

        embedding = None
        if not MESSAGE_EMBEDDING_MODEL_FAILED:
            try:
                if MESSAGE_EMBEDDING_MODEL is None:
                    from sentence_transformers import SentenceTransformer
                    MESSAGE_EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                embedding_array = MESSAGE_EMBEDDING_MODEL.encode(
                    text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embedding = torch.tensor(embedding_array, dtype=torch.float32)
            except Exception:
                MESSAGE_EMBEDDING_MODEL_FAILED = True
                self.message_embedding_model_failed = True

        if embedding is None:
            embedding = self._hashed_sentence_embedding(text)

        if embedding.numel() != self.message_embedding_dim:
            fixed_embedding = torch.zeros(self.message_embedding_dim, dtype=torch.float32)
            copy_len = min(self.message_embedding_dim, embedding.numel())
            fixed_embedding[:copy_len] = embedding[:copy_len]
            embedding = fixed_embedding
        self.message_embedding_cache[cache_key] = embedding.detach().cpu()
        return embedding.clone()

    def _hashed_sentence_embedding(self, text: str) -> torch.Tensor:
        embedding = torch.zeros(self.message_embedding_dim, dtype=torch.float32)
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
            token_hash = int.from_bytes(digest, byteorder="little", signed=False)
            index = token_hash % self.message_embedding_dim
            sign = 1.0 if ((token_hash >> 8) & 1) else -1.0
            embedding[index] += sign
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _task_text(self, inputs: Any) -> str:
        if isinstance(inputs, dict):
            for key in ("task", "question", "prompt", "input"):
                if key in inputs and inputs[key] is not None:
                    return str(inputs[key])
        return str(inputs or "")

    def _message_context_text(self, inputs: Any, predecessor: Node, receiver: Node, output: Any) -> str:
        sender_role = predecessor.role or predecessor.__class__.__name__
        receiver_role = receiver.role or receiver.__class__.__name__
        return (
            "Task:\n"
            f"{self._task_text(inputs)}\n\n"
            "Sender role:\n"
            f"{sender_role}\n\n"
            "Receiver role:\n"
            f"{receiver_role}\n\n"
            "Message:\n"
            f"{output}"
        )

    def _message_context_input(self, inputs: Any, predecessor: Node, receiver: Node, output: Any) -> torch.Tensor:
        context_text = self._message_context_text(inputs, predecessor, receiver, output)
        embedding = self._sentence_embedding(context_text)
        return embedding.to(self.message_complexity_action_bias.device)

    def _edge_idx_for_message(self, edge_kind: str, predecessor: Node, receiver: Node):
        if edge_kind == "spatial":
            return self.spatial_edge_index.get((predecessor.id, receiver.id))
        if edge_kind == "temporal":
            return self.temporal_edge_index.get((predecessor.id, receiver.id))
        return None

    def apply_contextual_message_selection(self, receiver: Node, inputs: Any, round: int, task_complexity: float):
        if self._active_message_selection_policy() != "aims_plus":
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)

        candidates = self._incoming_message_candidates(receiver)
        if not candidates:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)

        candidate_entries = []
        for edge_kind, predecessor, output in candidates:
            edge_idx = self._edge_idx_for_message(edge_kind, predecessor, receiver)
            if edge_idx is None:
                continue
            context_input = self._message_context_input(inputs, predecessor, receiver, output)
            candidate_entries.append({
                "edge_kind": edge_kind,
                "predecessor": predecessor,
                "edge_idx": edge_idx,
                "context_features": context_input,
            })

        if not candidate_entries:
            return torch.tensor(0.0, device=self.message_complexity_action_bias.device)

        context_features = torch.stack([
            candidate_entry["context_features"]
            for candidate_entry in candidate_entries
        ])
        gat_action_logits = self.message_action_gat(context_features)

        log_probs = [
            torch.tensor(
                0.0,
                device=self.message_complexity_action_bias.device,
                requires_grad=self.optimized_spatial_message_type or self.optimized_temporal_message_type,
            )
        ]
        for candidate_idx, candidate_entry in enumerate(candidate_entries):
            edge_kind = candidate_entry["edge_kind"]
            predecessor = candidate_entry["predecessor"]
            action_logits = self._message_type_logits_for(
                edge_kind,
                round,
            )[candidate_entry["edge_idx"]]
            action_logits = action_logits + gat_action_logits[candidate_idx].to(action_logits.device)
            action_logits = action_logits + self.message_complexity_action_bias.to(action_logits.device) * float(task_complexity)
            action_logits = action_logits / self.message_selection_temperature
            self._record_message_action_probs(torch.softmax(action_logits, dim=-1))
            message_type, message_type_log_prob = self.sample_message_type(
                edge_kind=edge_kind,
                edge_idx=candidate_entry["edge_idx"],
                round=round,
                return_log_prob=True,
                task_complexity=task_complexity,
                contextual_action_logits=gat_action_logits[candidate_idx],
            )
            if edge_kind == "spatial":
                predecessor.spatial_message_types[receiver.id] = message_type
            else:
                predecessor.temporal_message_types[receiver.id] = message_type
            self._record_message_action(edge_kind, message_type)
            log_probs.append(message_type_log_prob)

        return torch.sum(torch.stack(log_probs))

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)



    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None, task_complexity: float = 0.0,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial or self.optimized_spatial_message_type)]
        
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            # print(out_node)
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                    out_node.spatial_message_types[in_node.id] = self.sample_message_type("spatial", edge_idx, task_complexity=task_complexity)
                    # print(potential_connection)
                continue
            if not self.check_cycle(in_node, {out_node}):
                # print("@@@@@@@@@@@")
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                    message_type, message_type_log_prob = self.sample_message_type("spatial", edge_idx, return_log_prob=True, task_complexity=task_complexity)
                    out_node.spatial_message_types[in_node.id] = message_type
                    log_probs.append(message_type_log_prob)
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    
    
    def construct_spatial_connection_diff(self, round:int = 0, temperature: float = 1.0, threshold: float = None, task_complexity: float = 0.0,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial or self.optimized_spatial_message_type)]
        
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_spatial_edges, self.spatial_logits[round], self.spatial_masks[round])):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            out_id = list(self.nodes).index(out_node.id)
            in_id = list(self.nodes).index(in_node.id)
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                # if round == self.rounds-1 and self.dec_1 and (self.decision_masks[in_id]==0 or self.decision_masks[out_id]==0):
                #     print(11111)
                #     continue
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                    out_node.spatial_message_types[in_node.id] = self.sample_message_type("spatial", edge_idx, round=round, task_complexity=task_complexity)
                    # print(potential_connection)
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                    message_type, message_type_log_prob = self.sample_message_type("spatial", edge_idx, round=round, return_log_prob=True, task_complexity=task_complexity)
                    out_node.spatial_message_types[in_node.id] = message_type
                    log_probs.append(message_type_log_prob)
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None, task_complexity: float = 0.0,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal or self.optimized_temporal_message_type)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                    out_node.temporal_message_types[in_node.id] = self.sample_message_type("temporal", edge_idx, task_complexity=task_complexity)
                    # print(potential_connection)
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
                message_type, message_type_log_prob = self.sample_message_type("temporal", edge_idx, return_log_prob=True, task_complexity=task_complexity)
                out_node.temporal_message_types[in_node.id] = message_type
                log_probs.append(message_type_log_prob)
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def construct_temporal_connection_diff(self, round:int = 0, temperature: float = 1.0, threshold: float = None, task_complexity: float = 0.0,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal or self.optimized_temporal_message_type)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for edge_idx, (potential_connection, edge_logit, edge_mask) in enumerate(zip(self.potential_temporal_edges, self.temporal_logits[round-1], self.temporal_masks[round-1])):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                    out_node.temporal_message_types[in_node.id] = self.sample_message_type("temporal", edge_idx, round=round, task_complexity=task_complexity)
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
                message_type, message_type_log_prob = self.sample_message_type("temporal", edge_idx, round=round, return_log_prob=True, task_complexity=task_complexity)
                out_node.temporal_message_types[in_node.id] = message_type
                log_probs.append(message_type_log_prob)
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        self.reset_message_action_counts()
        log_probs = 0
        task_complexity = self._task_complexity(inputs)
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection(task_complexity=task_complexity)
            log_probs += self.construct_temporal_connection(round, task_complexity=task_complexity)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                log_probs += self.apply_contextual_message_selection(
                    self.nodes[current_node_id],
                    inputs,
                    round,
                    task_complexity,
                )
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        self._print_message_action_counts()
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 6000,
                  skip: bool=False,
                  case: bool=False) -> List[Any]:
        # inputs:{'task':"xxx"}
        self.reset_message_action_counts()
        log_probs = 0
        log_probs_skip = 0
        all_answers = []
        task_complexity = self._task_complexity(input)
        for round in range(num_rounds):
            round_answers = {}
            if not self.diff:
                log_probs += self.construct_spatial_connection(task_complexity=task_complexity)
                log_probs += self.construct_temporal_connection(round, task_complexity=task_complexity)
            else:
                log_probs += self.construct_spatial_connection_diff(round, task_complexity=task_complexity)
                log_probs += self.construct_temporal_connection_diff(round, task_complexity=task_complexity)
            
            # print(self.num_edges)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            in_degree_t = {node_id: len(node.temporal_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue_t = [node_id for node_id, deg in in_degree_t.items() if deg == 0]

            selected_index=-1

            if round <= 5 and skip:
                # log_probs = 0
                min_logit=100
                min_node=None
                min_loss=100
                loss_t_list = []
                loss_f_list = []
                log_list=[]
                for node_id, node in self.nodes.items():
                    in_id = list(self.nodes).index(node_id)
                    count=0
                    logits_count=0.
                    loss_t=0
                    loss_f=0
                    t=1.0
                    
                    for last_node in node.spatial_successors:
                        last_id = list(self.nodes).index(last_node.id)
                        count+=1
                        # logits_count+=torch.sigmoid(t*self.spatial_logits_1[round][in_id*5+last_id])
                        logits_count+=t*self.spatial_logits_1[round][in_id*5+last_id]
                        loss_t+=torch.log(1-torch.sigmoid(self.spatial_logits_1[round][in_id*5+last_id]))
                        loss_f+=torch.log(torch.sigmoid(self.spatial_logits_1[round][in_id*5+last_id]))
                    for last_node in node.spatial_predecessors:
                        last_id = list(self.nodes).index(last_node.id)
                        count+=1
                        # logits_count+=torch.sigmoid(t*self.spatial_logits_1[round][last_id*5+in_id])
                        logits_count+=t*self.spatial_logits_1[round][last_id*5+in_id]
                        loss_t+=torch.log(1-torch.sigmoid(self.spatial_logits_1[round][last_id*5+in_id]))
                        loss_f+=torch.log(torch.sigmoid(self.spatial_logits_1[round][last_id*5+in_id]))
                    # for last_node in node.temporal_predecessors:
                    #     last_id = list(self.nodes).index(last_node.id)
                    #     count+=1
                    #     # logits_count+=torch.sigmoid(t*self.temporal_logits_1[round-1][last_id*5+in_id])
                    #     logits_count+=t*self.temporal_logits_1[round-1][last_id*5+in_id]
                    #     loss_t+=torch.log(1-torch.sigmoid(self.temporal_logits_1[round-1][last_id*5+in_id]))
                    #     loss_f+=torch.log(torch.sigmoid(self.temporal_logits_1[round-1][last_id*5+in_id]))
                    log_list.append(logits_count)
                    if count==0:
                        count=1.0
                    loss_t_list.append(loss_t)
                    loss_f_list.append(loss_f)
                    # if logits_count / count < min_logit:
                    #     min_logit = logits_count / count
                    #     min_avg = logits_count / count
                    #     min_node = node_id
                    # elif logits_count / count == min_logit:
                    #     if len(node.spatial_predecessors)+len(node.temporal_predecessors)>len(self.find_node(min_node).spatial_predecessors)+len(self.find_node(min_node).temporal_predecessors):
                    #         # print(1111)
                    #         min_node = node_id
                    
                p = torch.softmax(torch.tensor(log_list),dim=0)
                selected_index = torch.multinomial(p, num_samples=1, replacement=False)
                # selected_index = list(self.nodes).index(min_node)
                # selected_index = 1
                for i in range(5):
                    if i==selected_index:
                        log_probs_skip += 4.0*loss_t_list[i]
                    else:
                        log_probs_skip += loss_f_list[i]
                # print(selected_index)
                # if self.skip_nodes:
                #     self.skip_nodes[round] = random.randint(0, 4)

            # for i in range(len(self.skip_nodes)):
            #     self.skip_nodes[i] = random.randint(0, 4)
            while zero_in_degree_queue:
                # if round==1 and i<2:
                #     continue
                current_node_id = zero_in_degree_queue.pop(0)
                log_probs += self.apply_contextual_message_selection(
                    self.nodes[current_node_id],
                    input,
                    round,
                    task_complexity,
                )
                tries = 0
                while tries < max_tries:
                    try:
                        # if current_node_id in need_skip:
                        if list(self.nodes).index(current_node_id) == selected_index and skip:
                            # print(111)
                            # if selected_index==1:
                            self.find_node(current_node_id).outputs = ['None.']
                            break
                        elif self.skip_nodes:
                            if list(self.nodes).index(current_node_id) == self.skip_nodes[round]:
                                self.find_node(current_node_id).outputs = ['None.']
                                break
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        # print(self.find_node(current_node_id).outputs)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            for node in self.nodes:
                round_answers[self.nodes[node].role+str(node)] = self.nodes[node].outputs
            all_answers.append(round_answers)
            self.update_memory()
        
        # if self.dec_1==False:
        if len(self.potential_spatial_edges)>0:
            self.connect_decision_node()
            await self.decision_node.async_execute(input)
            final_answers = self.decision_node.outputs
        else:
            final_answers = list(self.nodes.values())[0].outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        self._print_message_action_counts()
        # print(log_probs)
        # if skip:
        #     return final_answers, selected_index
        # else:
        if skip:
            return final_answers, log_probs_skip
        elif case:
            return final_answers, log_probs, all_answers
        else:
            return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
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
    
    def update_masks_diff(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            for i in range(self.rounds):
                num_edges = (self.spatial_masks[i] > 0).sum()
                num_masks = (self.spatial_masks[i] == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
                _edge_logits = self.spatial_logits[i].clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.spatial_masks[i] == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                # _edge_logits = torch.cat([_edge_logits[:num_masks], _edge_logits[num_masks:][torch.randperm(len(_edge_logits)-num_masks)]])
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.spatial_masks[i][prune_idx] = 0
        
        if self.optimized_temporal:
            for i in range(self.rounds-1):
                num_edges = (self.temporal_masks[i] > 0).sum()
                num_masks = (self.temporal_masks[i] == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
                _edge_logits = self.temporal_logits[i].clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.temporal_masks[i] == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                # _edge_logits = torch.cat([_edge_logits[:num_masks], _edge_logits[num_masks:][torch.randperm(len(_edge_logits)-num_masks)]])
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.temporal_masks[i][prune_idx] = 0
        # self.spatial_logits_1 = self.spatial_logits.clone()
        # self.temporal_logits_1 = self.temporal_logits.clone()
        return self.spatial_masks, self.temporal_masks

    def update_masks_dec(self):
        spatial_matrix_train = [param.reshape((5, 5)) for param in self.spatial_logits_1]
        temporal_matrix_train = [param.reshape((5, 5)) for param in self.temporal_logits_1]
        # spatial_mask_train = [param.reshape((5, 5)) for param in self.spatial_masks]
        # temporal_mask_train = [param.reshape((5, 5)) for param in self.temporal_masks]

        for i in range(len(spatial_matrix_train)):
            min = 100
            min_node = -1
            for j in range(5):
                sum = torch.sum(spatial_matrix_train[i][j,:]).item() + torch.sum(spatial_matrix_train[i][:,j]).item()
                # if i >= 1:
                #     sum += torch.sum(temporal_matrix_train[i-1][j,:]).item() + torch.sum(temporal_matrix_train[i-1][:,j]).item()
                count = torch.sum(self.fixed_spatial_masks[j,:]).item() + torch.sum(self.fixed_spatial_masks[:,j]).item()
                sum = sum / count
                if sum < min:
                    min = sum
                    min_node = j
            # min_node=random.randint(0, 4)
            self.skip_nodes.append(min_node)
            for k in range(5):
                self.spatial_masks[i][min_node*5+k]=0
                self.spatial_masks[i][k*5+min_node]=0
            if i > 0:
                for k in range(5):
                    self.temporal_masks[i-1][k*5+min_node]=0
            if i < len(spatial_matrix_train) - 1:
                for k in range(5):
                    self.temporal_masks[i][min_node*5+k]=0
