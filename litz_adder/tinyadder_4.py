#!/usr/bin/env python3
"""
4-digit version of TinyAdder: a hand-crafted transformer for 4-digit addition.

Adapted from tinyadder_module.py (10-digit version) by separating NUM_DIGITS
(digits per input number) from NUM_CANDIDATES (possible output digits, always 10
for base-10 arithmetic). The core algorithm is identical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, exp

# === Constants ===
NUM_DIGITS = 4        # number of digits in numbers to add
NUM_CANDIDATES = 10   # possible output digits (0-9), always 10 for base-10
TOKENS = [str(i) for i in range(10)] + ["=", "<bos>", "<eos>", "+"]

DIGIT_EMBED_SCALE = 10
V_SCALE = 1e4
DIGIT_SCALE = 10 ** NUM_DIGITS   # 1e4 for 4-digit addition
FINAL_SCALE = 100
DIGIT_OFFSET = 0.5
GATE_BIAS_SHIFT = 15.0
ALIBI_CONSTANT = log(10)

EQ_DIM, SPECIAL_DIM, DIGIT_DIM, COUNT_DIM, SCALE_DIM = 0, 1, 2, 3, 4
EMBEDDING_DIM = 5
LAYER0_HEADS = 5
ADJUSTMENT_HEAD = 3
SCALE_HEAD = 4
CANDIDATES_START = 5
DIGIT_POS_DIM = CANDIDATES_START + NUM_CANDIDATES     # slot for place-value accumulator
LAYER1_D_MODEL = CANDIDATES_START + NUM_CANDIDATES + 1

K_DIGIT_SCORE = -1000.0
K_SPECIAL_SCORE = -40.0
V_PROJ_SPECIAL = 0.1
V_PROJ_NEG_DOUBLE = -1.1
V_PROJ_SCALE = exp(K_SPECIAL_SCORE - log(10))

verbose = 0
_context_tokens = None


def _dump(label, t):
	"""Print a tensor with one row per sequence position, rounded to 3 decimals."""
	if verbose < 2:
		return
	print(f"\n--- {label} {list(t.shape[1:])} ---")
	vals = t[0]
	for i, row in enumerate(vals):
		tok = TOKENS[_context_tokens[i]] if _context_tokens and i < len(_context_tokens) else "?"
		val = row.tolist()
		if isinstance(val, (int, float)):
			nums = f"{val:.3f}" if isinstance(val, float) else str(val)
		else:
			nums = "  ".join(f"{v:.3f}" for v in val)
		print(f"  {i:2d} {tok:>5s} | {nums}")
	input("(press Enter)")


def softmax1(x, dim=-1):
	"""Softmax with a +1 in the denominator, allowing 'attend to nothing'."""
	exp_x = x.exp()
	return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


def apply_alibi(seq_len, n_heads):
	pos = torch.arange(seq_len)
	rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
	slopes = torch.zeros(n_heads, dtype=torch.float64)
	slopes[ADJUSTMENT_HEAD] = ALIBI_CONSTANT
	return slopes.unsqueeze(1).unsqueeze(2) * rel_pos.unsqueeze(0)


def pad_to(x, d):
	if x.size(-1) >= d:
		return x[..., :d]
	return torch.cat([x, torch.zeros(*x.shape[:-1], d - x.size(-1), dtype=x.dtype)], dim=-1)


# ---------------------------------------------------------------------------
# Module components
# ---------------------------------------------------------------------------

class SparseEmbedding(nn.Module):
	"""Embedding: 14 tokens -> 5 dims, with only 13 non-zero values."""

	def __init__(self):
		super().__init__()
		emb_idx = [[i, DIGIT_DIM] for i in range(1, 10)]
		emb_idx += [[10, EQ_DIM], [10, SPECIAL_DIM], [11, SPECIAL_DIM], [13, SPECIAL_DIM]]
		emb_val = [float(i * DIGIT_EMBED_SCALE) for i in range(1, 10)] + [1.0, 1.0, 1.0, 1.0]
		weight = torch.sparse_coo_tensor(
			torch.tensor(emb_idx).T,
			torch.tensor(emb_val, dtype=torch.float64),
			(14, 5),
		).to_dense()
		self.register_buffer("weight", weight)

	def forward(self, x):
		return self.weight[x]


class Layer0Attention(nn.Module):
	"""Layer 0 multi-head attention: 5 heads, d_k=1, with ALiBi and softmax1.

	Params: q bias (1 broadcast) + k weight,bias (2) + v weights (3) = 6
	"""

	def __init__(self):
		super().__init__()
		d = torch.float64
		self.q_bias = nn.Parameter(torch.ones(1, dtype=d))
		self.k_weight = nn.Parameter(torch.tensor(K_SPECIAL_SCORE - K_DIGIT_SCORE, dtype=d))
		self.k_bias = nn.Parameter(torch.tensor(K_DIGIT_SCORE, dtype=d))
		self.v_w1 = nn.Parameter(torch.tensor(V_PROJ_SPECIAL / V_PROJ_SCALE, dtype=d))
		self.v_w2 = nn.Parameter(torch.tensor(V_PROJ_NEG_DOUBLE / V_PROJ_SCALE, dtype=d))
		self.v_w3 = nn.Parameter(torch.tensor(1.0, dtype=d))

	def forward(self, h):
		B, S, _ = h.shape
		d = torch.float64

		q = torch.ones(B, S, LAYER0_HEADS, dtype=d, device=h.device) * self.q_bias

		k = torch.zeros(B, S, LAYER0_HEADS, dtype=d, device=h.device)
		k[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.k_weight + self.k_bias

		v = torch.zeros(B, S, LAYER0_HEADS, dtype=d, device=h.device)
		v[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.v_w1 + h[..., EQ_DIM] * self.v_w2
		v[..., SCALE_HEAD] = h[..., EQ_DIM] * self.v_w3

		q = q.view(B, S, LAYER0_HEADS, 1).transpose(1, 2)
		k = k.view(B, S, LAYER0_HEADS, 1).transpose(1, 2)
		v = v.view(B, S, LAYER0_HEADS, 1).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) + apply_alibi(S, LAYER0_HEADS).unsqueeze(0)
		scores = scores.masked_fill(torch.triu(torch.ones(S, S), 1).bool(), float('-inf'))
		attn = softmax1(scores, dim=-1).double()

		out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
		return out


class Layer0FFN(nn.Module):
	"""Layer 0 gated FFN: gate (1 broadcast) + up (NUM_CANDIDATES+1 values) + down (identity).

	Params: 1 + (NUM_CANDIDATES+1) = 12
	"""

	def __init__(self):
		super().__init__()
		pv = [(i + DIGIT_OFFSET) * DIGIT_SCALE * FINAL_SCALE for i in range(NUM_CANDIDATES)]
		self.up_vals = nn.Parameter(torch.tensor(pv + [DIGIT_SCALE], dtype=torch.float64))
		self.gate_weight = nn.Parameter(torch.ones(1, dtype=torch.float64))

	def forward(self, h):
		B, S, _ = h.shape
		d = torch.float64

		gate_in = torch.zeros(B, S, NUM_CANDIDATES + 1, dtype=d, device=h.device)
		gate_in[..., :NUM_CANDIDATES] = h[..., SCALE_DIM:SCALE_DIM + 1] * self.gate_weight
		gate_in[..., NUM_CANDIDATES] = h[..., DIGIT_DIM]
		gate_out = F.relu(gate_in)

		up_out = h[..., COUNT_DIM:COUNT_DIM + 1] * self.up_vals
		return gate_out * up_out


class Layer1Attention(nn.Module):
	"""Layer 1 attention: 1 head, Q=K=0 (uniform causal), V = linear projection + bias.

	Params: v weight (1) + v bias (1) = 2
	"""

	def __init__(self):
		super().__init__()
		d = torch.float64
		self.v_weight = nn.Parameter(torch.tensor(FINAL_SCALE, dtype=d))
		self.v_bias = nn.Parameter(torch.tensor(GATE_BIAS_SHIFT, dtype=d))

	def forward(self, h):
		B, S, _ = h.shape
		d = torch.float64

		q = torch.zeros(B, S, 1, dtype=d, device=h.device)
		k = torch.zeros(B, S, 1, dtype=d, device=h.device)

		v = h[..., DIGIT_POS_DIM:DIGIT_POS_DIM + 1] * self.v_weight + self.v_bias

		q = q.view(B, S, 1, 1).transpose(1, 2)
		k = k.view(B, S, 1, 1).transpose(1, 2)
		v = v.view(B, S, 1, 1).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1))
		scores = scores.masked_fill(torch.triu(torch.ones(S, S), 1).bool(), float('-inf'))
		attn = softmax1(scores, dim=-1).double()

		out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
		return out


class Layer1FFN(nn.Module):
	"""Layer 1 V-shaped FFN: relu(+x) + relu(-x) = |x|, scaled.

	Params: +V_SCALE (1) + -V_SCALE (1) + FINAL_SCALE broadcast (1) = 3
	"""

	def __init__(self):
		super().__init__()
		d = torch.float64
		self.gate_pos_scale = nn.Parameter(torch.tensor(V_SCALE, dtype=d))
		self.gate_neg_scale = nn.Parameter(torch.tensor(-V_SCALE, dtype=d))
		self.up_scale = nn.Parameter(torch.tensor(FINAL_SCALE, dtype=d))

	def forward(self, candidates):
		gate_pos = F.relu(candidates * self.gate_pos_scale)
		gate_neg = F.relu(candidates * self.gate_neg_scale)
		return (gate_pos + gate_neg) * self.up_scale


class TinyAdder4Module(nn.Module):
	"""
	36-parameter transformer for 4-digit addition, as nn.Module.

	Architecture: Embedding(13) -> Layer0[Attn(6) + FFN(12)] -> Layer1[Attn(2) + FFN(3)]
	Total: 13 + 6 + 12 + 2 + 3 = 36 parameters
	"""

	def __init__(self):
		super().__init__()
		self.embedding = SparseEmbedding()
		self.layer0_attn = Layer0Attention()
		self.layer0_ffn = Layer0FFN()
		self.layer1_attn = Layer1Attention()
		self.layer1_ffn = Layer1FFN()

	@torch.inference_mode()
	def forward(self, x):
		B, S = x.shape

		h = self.embedding(x)
		h = pad_to(h, EMBEDDING_DIM)
		_dump("After embedding", h)

		attn0_out = self.layer0_attn(h)
		_dump("Layer0 attn output", attn0_out)
		h = h + attn0_out
		_dump("After Layer0 attn + residual", h)

		ffn0_out = self.layer0_ffn(h)
		_dump("Layer0 FFN output", ffn0_out)
		h = pad_to(h, LAYER1_D_MODEL)
		h[..., CANDIDATES_START:LAYER1_D_MODEL] = h[..., CANDIDATES_START:LAYER1_D_MODEL] + ffn0_out
		_dump("After Layer0 FFN + residual (widened)", h)

		attn1_out = self.layer1_attn(h)
		_dump("Layer1 attn output", attn1_out)
		h = h + attn1_out
		_dump("After Layer1 attn + residual", h)

		candidates = h[..., CANDIDATES_START:CANDIDATES_START + NUM_CANDIDATES]
		_dump("Candidates (pre-FFN)", candidates)
		ffn1_out = self.layer1_ffn(candidates)
		_dump("Layer1 FFN output", ffn1_out)
		h = pad_to(h, NUM_CANDIDATES)
		h = h + ffn1_out
		_dump("Final h (argmin input)", h)

		result = h.argmin(dim=-1)
		_dump("Result (indexes minimizing h)", result)
		return result


def add(model, a, b):
	"""Use the model to compute a + b autoregressively."""
	S = f"{a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}="
	generated = []
	for _ in range(NUM_DIGITS + 1):
		toks = [TOKENS.index(t) for t in ["<bos>"] + list(S)]
		context = ''.join(TOKENS[t] for t in toks)
		global _context_tokens
		_context_tokens = toks
		x = torch.tensor(toks).unsqueeze(0)
		pred = model.forward(x)
		next_digit = TOKENS[int(pred[0, -1].item())]
		S += next_digit
		if verbose >= 1:
			pred_str = ''.join(TOKENS[t] for t in pred[0])
			print(f"context: {context}  --> prediction: {pred_str}")
		generated.append(next_digit)
	return int("".join(generated))


def self_test(n=100, seed=42):
	"""Run n random addition tests and print results."""
	import random
	random.seed(seed)
	model = TinyAdder4Module()
	correct = 0
	failures = []
	for _ in range(n):
		max_val = 10**NUM_DIGITS - 1
		a = random.randint(0, max_val)
		b = random.randint(0, max_val)
		result = add(model, a, b)
		expected = a + b
		if result == expected:
			correct += 1
		else:
			failures.append((a, b, expected, result))
	print(f"4-digit self-test: {correct}/{n}")
	if failures and len(failures) <= 10:
		for a, b, expected, result in failures:
			print(f"  FAIL: {a} + {b} = {expected}, got {result}")


if __name__ == "__main__":
	verbose = 0
	self_test()
	verbose = 2
	result = add(TinyAdder4Module(), 1234, 5678)
	print(f"Final result: {result}")
