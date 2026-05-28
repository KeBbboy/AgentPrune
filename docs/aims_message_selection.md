# AIMS/AIMS+: Adaptive Information-granularity Message Selection

This branch adds AIMS and AIMS+, a research-oriented message selection layer for
AgentPrune. The default behavior remains unchanged unless
`--message_selection_policy` is set, or the shorthand `--aims_plus` flag is
used.

## Motivation

The original graph optimizer prunes spatial-temporal communication edges. The
previous message selection hook was much weaker: each edge had one scalar logit
and, when enabled, could only choose `answer` or `reason`. AIMS makes message
selection a first-class policy over information granularity, so it can trade off
accuracy, redundancy, and token cost after the topology has been chosen.

The design follows three lines of prior work:

- AgentPrune formalizes redundant LLM-agent communication and learns a
  token-economic spatial-temporal topology: https://arxiv.org/abs/2410.02506
- Limited-bandwidth multi-agent communication work shows that adaptive gating can
  prune redundant messages while preserving or improving performance:
  https://arxiv.org/abs/1912.05304
- Adaptive Graph Pruning argues that useful multi-agent communication should be
  task-adaptive rather than fixed: https://arxiv.org/abs/2506.02951

## AIMS policy

For each candidate edge e = (sender, receiver), round r, and task q, AIMS samples
an action from:

```
A = {full, answer, reason}
p(a | e, r, q) = softmax((theta[e, r] + c(q) * beta) / tau)
```

where:

- `theta[e, r]` is a trainable categorical logit vector for the edge and round.
- `c(q)` is a deterministic task-complexity feature from prompt length and line
  count.
- `beta` is a trainable action bias that lets harder tasks prefer richer
  messages.
- `tau` is `--message_selection_temperature`.

The initialization is role-aware. For example, math-solver receivers start with a
small preference for `answer` hints, inspectors and analysts prefer `reason`, and
HumanEval/programming agents prefer `full` because code payloads are fragile.

## AIMS+ contextual per-edge policy

AIMS+ adds a second, content-aware per-edge policy that runs immediately before a
receiver agent consumes its incoming messages. This matters because the original
topology stage runs before the sender's actual output is known.

The AIMS+ action space is:

```
A+ = {answer, reason, full}
```

The contextual policy is:

```
z_i = Enc([q, role(sender_i), role(receiver), m_i])
g_i = Generator(z_i)
u_i = Head(g_i) + gamma * tanh(Critic([z_i, g_i]))
p(a_i | e_i, r, q, role(sender_i), role(receiver), m_i) =
    softmax((theta[e_i, r] + u_i + c(q) * beta) / tau)
```

where `m_i` is one generated sender message. Following GDesigner's lightweight
profile/query embedding design, AIMS+ encodes message text with
`sentence-transformers/all-MiniLM-L6-v2` when the model is available locally, and
falls back to a deterministic hash embedding otherwise. The encoder input for
each retained edge is the serialized tuple `[task, sender role, receiver role,
message]`, producing one 384-dimensional context embedding per edge.

At execution time, AIMS+ directly samples one action for each retained incoming
edge after observing the actual sender output. There is no hard receiver-level
budget constraint; token pressure comes from the expected message-cost
regularizer.

This gives the method three distinct layers:

1. Spatial-temporal topology pruning decides whether a communication channel
   exists.
2. Role-aware edge priors learn each edge's usual information granularity.
3. A content-aware receiver gate adapts each retained edge's payload after
   seeing the actual message.

The message-level action selector is implemented as a receiver-local GAN-style
generator/critic network over the incoming message embeddings. The generator
maps each embedding to an action-logit proposal, while the critic adds a small
bounded residual from the original embedding and generated feature. This avoids
the fully connected pairwise attention graph while keeping content-aware action
logits for every retained incoming edge. The remaining edge logits and
task-complexity bias serve as priors.

Since AgentPrune already handles edge-level pruning, AIMS+ does not include a
`none` action. Every retained edge transmits exactly one of `answer`, `reason`,
or `full`.

## Objective

AIMS extends the existing REINFORCE-style AgentPrune objective:

```
L = - (utility - lambda_r * selected_reason_count - lambda_f * selected_full_count)
      * log p(topology, message_actions)
    + graph_regularization
    + lambda_c * expected_message_cost
    - lambda_h * action_entropy
    - lambda_d * receiver_granularity_diversity
```

The regularizers are exposed as:

- `--message_cost_weight`
- `--message_entropy_weight`
- `--message_diversity_weight`
- `--message_context_weight_decay`
- `--message_reason_penalty_weight`
- `--message_full_penalty_weight`

The cost term rewards concise messages, the entropy term prevents early action
collapse, the receiver-side diversity term discourages all incoming messages from
using the same granularity, the sampled-action penalties directly discourage
overusing `reason` and especially `full`, and the context weight decay keeps the
GAN selector stable.

## Usage

Train and evaluate AIMS on a shared training path such as MMLU:

```bash
python experiments/run_mmlu.py \
  --mode FakeAGFull \
  --agent_nums 6 \
  --batch_size 4 \
  --num_iterations 10 \
  --imp_per_iterations 5 \
  --pruning_rate 0.25 \
  --num_rounds 1 \
  --optimized_spatial \
  --optimized_temporal \
  --aims_plus
```

Suggested ablations:

```bash
# Original AgentPrune topology only.
--message_selection_policy none

# Legacy two-action message selection.
--message_selection_policy binary

# AIMS without explicit token-cost pressure.
--message_selection_policy aims --message_cost_weight 0

# AIMS without receiver-side granularity diversity.
--message_selection_policy aims --message_diversity_weight 0

# AIMS+ with the contextual adapter strongly damped.
--message_selection_policy aims_plus --message_context_weight_decay 100
```

Report both task metric and token usage (`PromptTokens`, `CompletionTokens`) so
the method can be evaluated as an accuracy-cost Pareto improvement rather than
only as an accuracy change.
