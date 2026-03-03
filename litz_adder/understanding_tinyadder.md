# Understanding TinyAdder: A Step-by-Step Walkthrough

This document traces a complete forward pass of the TinyAdder model computing **1234 + 5678 = 6912**, examining the internal state after each step. The walkthrough covers the generation of the digit after "00000006" has already been produced (predicting "9").

The context window at this point is 31 tokens:
```
<bos>0000001234+0000005678=00000006
```

## Architecture Overview

TinyAdder is a hand-crafted 36-parameter, 2-layer transformer that performs 10-digit addition. Its architecture:

- **Embedding** (13 non-zero values): 14 tokens → 5 dims
- **Layer 0 Attention** (6 params): 5 heads, d_k=1, with ALiBi and softmax1
- **Layer 0 FFN** (12 params): gated FFN, widens from 5 to 11 dims
- **Layer 1 Attention** (2 params): 1 head, Q=K=0 (uniform causal)
- **Layer 1 FFN** (3 params): V-shaped absolute value via dual ReLU

The model uses a **residual stream** architecture: a fixed-width vector per position that each layer reads from and writes to. In standard transformers this width is constant; here it pragmatically widens from 5 → 16 → 10 to minimize parameter count.

---

## Step 1: Embedding

```
--- After embedding [31, 5] ---
   0 <bos> | 0.000  1.000  0.000  0.000  0.000
   1     0 | 0.000  0.000  0.000  0.000  0.000
   2     0 | 0.000  0.000  0.000  0.000  0.000
   ...
   7     1 | 0.000  0.000  10.000  0.000  0.000
   8     2 | 0.000  0.000  20.000  0.000  0.000
   9     3 | 0.000  0.000  30.000  0.000  0.000
  10     4 | 0.000  0.000  40.000  0.000  0.000
  11     + | 0.000  1.000  0.000  0.000  0.000
   ...
  18     5 | 0.000  0.000  50.000  0.000  0.000
  19     6 | 0.000  0.000  60.000  0.000  0.000
  20     7 | 0.000  0.000  70.000  0.000  0.000
  21     8 | 0.000  0.000  80.000  0.000  0.000
  22     = | 1.000  1.000  0.000  0.000  0.000
   ...
  30     6 | 0.000  0.000  60.000  0.000  0.000
```

`SparseEmbedding` maps each token to a 5-dimensional vector via a lookup table with only 13 non-zero values:

| Dim | Name | Purpose |
|-----|------|---------|
| 0 | EQ_DIM | 1.0 only for `=` |
| 1 | SPECIAL_DIM | 1.0 for `<bos>`, `+`, and `=` |
| 2 | DIGIT_DIM | digit value × 10 (so `0`→0, `1`→10, ..., `9`→90) |
| 3 | COUNT_DIM | always 0 (filled in by attention) |
| 4 | SCALE_DIM | always 0 (filled in by attention) |

The embedding is extremely sparse. Digits only activate dim 2 (with their value scaled by 10), special tokens activate dim 1, and `=` additionally activates dim 0. Token `0` maps to all zeros.

Dims 3 and 4 are "reserved lanes" — left at zero by the embedding for the attention layer to fill in. This is a consequence of the residual stream architecture: every layer reads from and writes to the same vector, so the width is determined by the maximum needs of any layer.

---

## Step 2: Layer 0 Attention

```
--- Layer0 attn output [31, 5] ---
   0 <bos> | 0.000  0.000  0.000  1.000  0.000
   1     0 | 0.000  0.000  0.000  0.100  0.000
   2     0 | 0.000  0.000  0.000  0.010  0.000
   3     0 | 0.000  0.000  0.000  0.001  0.000
   ...
  11     + | 0.000  0.000  0.000  1.000  0.000
  12     0 | 0.000  0.000  0.000  0.100  0.000
  13     0 | 0.000  0.000  0.000  0.010  0.000
  14     0 | 0.000  0.000  0.000  0.001  0.000
   ...
  22     = | 0.000  0.000  0.000  -10.000  0.042
  23     0 | 0.000  0.000  0.000  -1.000  0.040
  24     0 | 0.000  0.000  0.000  -0.100  0.038
   ...
  30     6 | 0.000  0.000  0.000  -0.000  0.031
```

Layer 0 has 5 attention heads (each with d_k=1, d_v=1), but only **2 do anything**: head 3 (ADJUSTMENT_HEAD → COUNT_DIM) and head 4 (SCALE_HEAD → SCALE_DIM). The number of heads must equal d_model because each head outputs a single value that maps to one dimension of the residual stream.

### Head 3: Positional Counter via ALiBi

This head has ALiBi (Attention with Linear Biases) slope = `log(10)` ≈ 2.3026. The key design:
- Special tokens (`<bos>`, `+`, `=`) get key score = **-40**
- All other tokens get key score = **-1000** (effectively invisible)

With `softmax1` (softmax with +1 in the denominator, allowing "attend to nothing"), all attention weights are tiny in absolute terms. But the ALiBi decay of `log(10)` per position means each step further from a special token gets **1/10th** the weight. The V values for special tokens are enormous (scaled by 1/V_PROJ_SCALE), so tiny_weight × huge_value produces clean decimal values.

The result is **a power-of-10 distance counter from the nearest special token**:
- Position 0 (`<bos>`): 1.000
- Position 1: 0.100 (1/10)
- Position 2: 0.010 (1/100)
- Position 3: 0.001 (1/1000)
- ...resets at `+` (position 11): 1.000, 0.100, 0.010, ...

After `=`, values go **negative** because `=` has `EQ_DIM=1`, which adds a negative V contribution (10× larger magnitude). So `=` at position 22 gives -10.000, then -1.000, -0.100, etc.

### Head 4: Output Gate Signal

This head has zero ALiBi slope, and V is non-zero only for `=` (via `EQ_DIM`). With Q=K=0, softmax1 gives uniform causal weights of `1/(n+2)`.

The output in SCALE_DIM is `1/(position+2)` for positions at or after `=`, and zero before:
- Position 22 (`=`): 1/24 ≈ 0.042
- Position 23: 1/25 = 0.040
- Position 30: 1/32 ≈ 0.031

This acts as a **gate signal** — it tells the Layer 0 FFN "you're in the answer section, activate the digit computation."

### Residual Add

Since the embedding had zeros in dims 3-4 and the attention had zeros in dims 0-2, they slot together cleanly:

```
--- After Layer0 attn + residual [31, 5] ---
  22     = | 1.000  1.000  0.000  -10.000  0.042
   ...
  30     6 | 0.000  0.000  60.000  -0.000  0.031
```

All five channels are now populated:
- **Dim 0 (EQ)**: is this the `=` token?
- **Dim 1 (SPECIAL)**: is this `<bos>`, `+`, or `=`?
- **Dim 2 (DIGIT)**: digit value × 10
- **Dim 3 (COUNT)**: power-of-10 distance from nearest special token (negative after `=`)
- **Dim 4 (SCALE)**: gate signal, non-zero only after `=`

---

## Step 3: Layer 0 FFN

```
--- Layer0 FFN output [31, 11] ---
   7     1 | 0  0  0  0  0  0  0  0  0  0  10000
   8     2 | 0  0  0  0  0  0  0  0  0  0  2000
   9     3 | 0  0  0  0  0  0  0  0  0  0  300
  10     4 | 0  0  0  0  0  0  0  0  0  0  40
   ...
  18     5 | 0  0  0  0  0  0  0  0  0  0  50000
  19     6 | 0  0  0  0  0  0  0  0  0  0  6000
  20     7 | 0  0  0  0  0  0  0  0  0  0  700
  21     8 | 0  0  0  0  0  0  0  0  0  0  80
  22     = | -208B  -625B  -1042B  -1458B  -1875B  -2292B  -2708B  -3125B  -3542B  -3958B  0
   ...
```

The FFN is a gated network: `output = ReLU(gate) × up`. It widens the output from 5 to 11 dimensions.

### Before `=`: Slot 10 Computes Place Values

Gate slot 10 reads DIGIT_DIM, and up reads COUNT_DIM × `1e10`. The math:

```
output[10] = ReLU(digit × 10) × (10^(-dist_from_special) × 1e10)
```

Position 7 (digit `1`, 7 steps from `<bos>`): `10 × 1e-7 × 1e10 = 10,000`
Position 8 (digit `2`): `20 × 1e-8 × 1e10 = 2,000`
Position 9 (digit `3`): `30 × 1e-9 × 1e10 = 300`
Position 10 (digit `4`): `40 × 1e-10 × 1e10 = 40`

Sum across first number's digits: 10000 + 2000 + 300 + 40 = **12,340** (1234 × 10, the extra factor from DIGIT_EMBED_SCALE). Same pattern gives **56,780** for the second number.

So slot 10 encodes each digit's **place-value contribution** using the power-of-10 counter from the attention layer.

### After `=`: Slots 0-9 Are Candidate Digit Scores

Now SCALE_DIM > 0, opening the gate for slots 0-9. Each candidate digit d gets a score proportional to `-(d + 0.5)`, scaled by COUNT and SCALE. At this stage, digit 9 would always win an argmin — the actual answer selection comes from Layer 1.

### Widening to 16 Dims

The residual stream is padded from 5 to 16 dims (`pad_to(h, LAYER1_D_MODEL)`), and the FFN's 11-dim output lands in slots 5-15:

- **Dims 0-4**: original residual (EQ, SPECIAL, DIGIT, COUNT, SCALE)
- **Dims 5-14**: candidate scores for digits 0-9
- **Dim 15**: place-value encoded digit contribution

The FFN internally computes in whatever width it wants — it just has to produce output that gets added to the residual. In standard transformers, FFNs typically expand to 4× width internally then project back down. Here, the author just widens the residual itself, since later layers need those extra dims.

---

## Step 4: Layer 1 Attention

```
--- Layer1 attn output [31, 1] ---
   0 <bos> | 7.500
   ...
   7     1 | 111124.444
   ...
  22     = | 288014.375
  23     0 | 276494.400
   ...
  30     6 | 28514.531
```

Layer 1 has a single head with Q=K=0, making softmax1 produce **uniform causal attention** — each position simply averages all V values from itself and everything before, with weight `1/(n+2)`.

V is computed from dim 15 (the place-value slot): `V = dim15 × 100 + 15`

### Computing the Running Sum

The output at each position is `sum(V_0..V_i) / (i+2)`. Verifying position 22 (`=`):

```
V sum of first number:  (10000+2000+300+40)×100 + 4×15 = 1,234,060
V sum of second number: (50000+6000+700+80)×100 + 4×15 = 5,678,060
V sum of 15 non-digit positions: 15 × 15            =         225
Total:                                                  6,912,345
Output: 6,912,345 / 24 = 288,014.375  ✓
```

The V sum encodes **(a + b) × 1000**, buried inside some bias noise.

### The Autoregressive Subtraction Trick

Position 30 (digit `6`, already generated) has dim 15 = -60,000, so V = -5,999,985. This **subtracts** from the running sum:

```
6,912,345 + 7×15 + (-5,999,985) = 912,465
Output: 912,465 / 32 = 28,514.531  ✓
```

Each previously generated answer digit subtracts its place-value contribution from the running total. Subsequent positions see the **remainder** — what's left to express as digits. The model is doing long addition with a running remainder, autoregressively.

### Broadcast Residual Add

The single output value gets broadcast-added to all 16 dims of h. This shifts all candidate scores (dims 5-14) by the same amount. The shift doesn't change which candidate wins argmin directly — instead, it positions the candidates relative to zero so the Layer 1 FFN's ReLU gate can do its work.

For position 30, after the shift, the candidates become:
```
digit 0:  26,952    digit 5:  11,327
digit 1:  23,827    digit 6:   8,202
digit 2:  20,702    digit 7:   5,077
digit 3:  17,577    digit 8:   1,952
digit 4:  14,452    digit 9:  -1,173
```

The attention shift was calibrated so that **digit 9 lands closest to zero** (slightly negative), while all others are pushed further away. The correct digit (9, since 1234+5678=6912 and we're predicting after "6") ends up nearest to zero.

---

## Step 5: Layer 1 FFN (V-Shaped Absolute Value)

```
--- Layer1 FFN output [31, 10] ---
   ...
  30     6 | 26.95B  23.83B  20.70B  17.58B  14.45B  11.33B  8.20B  5.08B  1.95B  1.17B
```

The FFN computes `|x| × V_SCALE × FINAL_SCALE` = `|x| × 1,000,000` using the dual-ReLU trick:
- `ReLU(x × V_SCALE)` keeps the positive part
- `ReLU(x × -V_SCALE)` keeps the negative part (flipped positive)
- Sum = `|x| × V_SCALE`, then multiply by FINAL_SCALE

There is no standard `|x|` activation in transformer toolkits, so the author decomposes absolute value into two ReLU operations — a structure that fits naturally into a 2-neuron gated FFN.

For position 30:
```
digit 0:  26,952  → |26,952| × 1e6  ≈ 26,952,031,252
digit 1:  23,827  → |23,827| × 1e6  ≈ 23,827,031,252
...
digit 8:   1,952  → | 1,952| × 1e6  ≈  1,952,031,252
digit 9:  -1,173  → | 1,173| × 1e6  ≈  1,172,968,748  ← smallest!
```

Digit 9 was closest to zero before the FFN, and the absolute value function makes "closest to zero" equivalent to "smallest value." Argmin picks digit **9**.

---

## Step 6: Final Residual Add and Argmin

The residual stream is truncated from 16 to 10 dims (`pad_to(h, 10)`), then the FFN output is added. The residual content (single-digit to low-thousands) is negligible compared to the FFN output (billions), so the argmin is entirely determined by the FFN.

```
--- Final h (argmin input) [31, 10] ---
  30     6 | 26.95B  23.83B  20.70B  17.58B  14.45B  11.33B  8.20B  5.08B  1.95B  1.17B
                                                                                     ↑ min
```

`h.argmin(dim=-1)` selects dim 9 → digit **9**. This is appended to the context, and the process repeats for the next digit.

---

## Summary: How TinyAdder Adds

1. **Embedding**: Encodes digit values (×10) and marks special tokens (`<bos>`, `+`, `=`)

2. **Layer 0 Attention**: Two active heads:
   - Head 3 creates a **power-of-10 position counter** using ALiBi with slope log(10)
   - Head 4 creates a **gate signal** that's non-zero only after `=`

3. **Layer 0 FFN**: Two functions:
   - **Before `=`**: Computes place-value contributions (digit × 10^place) using the position counter
   - **After `=`**: Creates candidate digit scores (an evenly-spaced negative ramp)

4. **Layer 1 Attention**: Uniform causal averaging sums all place-value contributions, giving a scaled encoding of (a + b). Previously generated answer digits subtract from this sum, creating a **running remainder**.

5. **Layer 1 FFN**: V-shaped `|x|` function (via dual ReLU) converts "closest to zero" into "smallest value"

6. **Argmin**: Picks the digit whose candidate score is smallest — the one the attention layer positioned closest to zero.

The overall mechanism is remarkably similar to how humans do long addition: sum the numbers, then extract digits from the result one at a time, maintaining a running remainder. The 36 parameters encode this algorithm entirely through the standard transformer primitives of attention, gating, and residual connections.
