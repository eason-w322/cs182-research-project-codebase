"""A routed security expert for OLMoE: one extra expert and one extra router score per block.

Written against transformers 5.23.0 (see reference/olmoe_moe_parts.py).

Usage:
    model = OlmoeForCausalLM.from_pretrained(...)
    blocks = add_security_experts(model)        # freezes the model, wraps all 16 blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP


class SecurityMoeBlock(nn.Module):
    """Replaces one OlmoeSparseMoeBlock (a decoder layer's `mlp`).

    Frozen:    the original router weights and the 64 original experts (reused as they are).
    Trainable: the new expert (an OlmoeMLP), its router row and its router bias.
    """

    def __init__(self, original_block, config):
        super().__init__()

        # 1. Keep the original block: its router (`gate`) and its 64 experts (`experts`).
        self.original = original_block

        # 2. Freeze it: none of its weights will be trained.
        for p in self.original.parameters():
            p.requires_grad = False

        # 3. Sizes, read from the original router.
        self.num_experts = original_block.gate.num_experts   # 64; also the index of the new expert
        self.top_k = original_block.gate.top_k               # 8
        self.hidden_size = original_block.gate.hidden_dim    # 2048

        # 4. The new expert: same shape and math as each original expert.
        self.new_expert = OlmoeMLP(config)

        # 5. Its output starts at exactly zero, so at first it changes nothing when selected.
        nn.init.zeros_(self.new_expert.down_proj.weight)

        # 6. The new expert's router row: its score for a token is (token . router_row) + router_bias.
        self.router_row = nn.Parameter(torch.zeros(self.hidden_size))

        # 7. A single learnable number added to that score; sets how often the expert is picked overall.
        self.router_bias = nn.Parameter(torch.zeros(()))

        # 8. masked=True: behave exactly like the original block (for checks and for the reference terms).
        self.masked = False

        # Filled by every unmasked forward pass, for the routing analysis: (B, T) bool,
        # True where the token's top-k included the new expert.
        self.last_selected = None

    def forward(self, hidden_states):
        if self.masked:
            return self.original(hidden_states)

        # Flatten: (B, T, H) -> (N, H), N = B * T tokens.
        batch_size, seq_len, hidden = hidden_states.shape
        h = hidden_states.reshape(-1, hidden)

        # Scores: 64 frozen ones from the original router, plus one new one.
        old_logits = F.linear(h, self.original.gate.weight)                                 # (N, 64)
        new_logit = h @ self.router_row.to(h.dtype) + self.router_bias.to(h.dtype)          # (N,)
        logits = torch.cat([old_logits, new_logit.unsqueeze(-1)], dim=-1)                   # (N, 65)

        # Same rule as the original router: softmax over all scores, keep the top k.
        probs = F.softmax(logits, dim=-1, dtype=torch.float)                                # (N, 65)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)                                # (N, k), (N, k)
        if self.original.gate.norm_topk_prob:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        top_w = top_w.to(h.dtype)

        # Which slots picked the new expert (index 64).
        picked = top_i == self.num_experts                                                  # (N, k) bool

        # The 64 original experts. The original experts module can't take index 64: its fast GPU
        # versions (e.g. "grouped_mm") don't skip it and would add garbage. So in those slots we
        # pass expert 0 with a share of exactly 0, which contributes nothing.
        orig_i = top_i.masked_fill(picked, 0)
        orig_w = top_w.masked_fill(picked, 0.0)
        out = self.original.experts(h, orig_i, orig_w)                                      # (N, H)

        # The new expert, on the tokens that picked it.
        token_idx, slot = picked.nonzero(as_tuple=True)
        if token_idx.numel() > 0:
            x = h[token_idx].to(self.new_expert.down_proj.weight.dtype)                     # its tokens
            y = self.new_expert(x)                                                          # (m, H)
            y = y * top_w[token_idx, slot, None]                                            # scale by share
            out = out.index_add(0, token_idx, y.to(out.dtype))                              # add into rows

        self.last_selected = picked.any(dim=-1).reshape(batch_size, seq_len).detach()
        return out.reshape(batch_size, seq_len, hidden)


def add_security_experts(model, layers=None):
    """Freeze the whole model, then wrap the MoE block of each chosen layer (default: all).

    Returns {layer index: SecurityMoeBlock}.
    """
    for p in model.parameters():
        p.requires_grad = False
    if layers is None:
        layers = range(len(model.model.layers))
    blocks = {}
    for i in layers:
        layer = model.model.layers[i]
        block = SecurityMoeBlock(layer.mlp, model.config)
        block.to(device=layer.mlp.gate.weight.device)
        layer.mlp = block
        blocks[i] = block
    return blocks


def trainable_parameters(model):
    """Only the new experts, router rows and biases."""
    return [p for p in model.parameters() if p.requires_grad]


def set_masked(model, masked: bool):
    """Turn every security expert off (True: exact original model) or on (False)."""
    for layer in model.model.layers:
        if isinstance(layer.mlp, SecurityMoeBlock):
            layer.mlp.masked = masked
