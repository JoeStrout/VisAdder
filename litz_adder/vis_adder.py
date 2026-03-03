#!/usr/bin/env python3
"""
Interactive zoomable visualization of the TinyAdder transformer.

Usage: python vis_adder.py
Controls:
  Mouse wheel     - zoom in/out (toward cursor)
  Mouse drag      - pan
  Left/Right keys - change autoregressive step (0-10)
  Click input box - edit A+B, Enter to recompute
  Tab             - toggle input box focus
  Esc             - quit
"""

import os
import pyray as rl
import torch
from dataclasses import dataclass
from tinyadder_module import (
    TinyAdderModule, TOKENS, NUM_DIGITS,
    pad_to, EMBEDDING_DIM, LAYER1_D_MODEL, CANDIDATES_START, DIGIT_POS_DIM,
)

# ── Font paths (relative to this script) ─────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
MONO_FONT_PATH = os.path.join(_DIR, "ProFontIIx.ttf")
TEXT_FONT_PATH = os.path.join(_DIR, "NotoSans-Regular.ttf")
FONT_SIZE = 32  # base size to load (rendered at various sizes via draw_text_ex)

# Globals set after window init
font_mono = None  # ProFontIIx — numbers, cell values, row labels
font_text = None  # NotoSans — block titles, annotations, UI text

# ── Display constants ──────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1400, 900
CELL_SIZE = 40.0        # world-space pixels per heatmap cell
BLOCK_GAP = 220       # vertical gap between tensor blocks (fits annotations)
ROW_LABEL_W = 80.0      # space reserved for row labels (left of heatmap)
ANNOTATION_X_OFF = 30.0 # annotation offset right of heatmap

# Zoom-level thresholds (screen pixels per cell)
LEVEL2_THRESH = 8       # column headers, shape
LEVEL3_THRESH = 18      # numeric values, row labels
LEVEL4_THRESH = 40      # explanatory annotations

# Colors
BG_COLOR = rl.Color(30, 30, 35, 255)
LABEL_COLOR = rl.Color(220, 220, 220, 255)
DIM_LABEL_COLOR = rl.Color(180, 180, 200, 255)
ROW_LABEL_COLOR = rl.Color(160, 160, 170, 255)
ANNOTATION_COLOR = rl.Color(255, 220, 100, 255)
ARROW_COLOR = rl.Color(80, 80, 90, 255)
HIGHLIGHT_COLOR = rl.Color(0, 255, 120, 255)


# ── Number formatters ─────────────────────────────────────────────────────
MAX_CHARS = 6  # hard limit on formatted value width

def _fmt_suffixed(v, suffixes=((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))):
    """Format a number with K/M/B/T suffix, strictly <= MAX_CHARS characters."""
    if abs(v) < 0.5:
        return "0"
    sign = "-" if v < 0 else ""
    a = abs(v)
    # Pick the right suffix
    for thresh, suf in suffixes:
        if a >= thresh:
            scaled = a / thresh
            # Budget: MAX_CHARS - len(sign) - len(suf)
            budget = MAX_CHARS - len(sign) - len(suf)
            # Try with decimals first, then integer
            for dec in range(2, -1, -1):
                s = f"{scaled:.{dec}f}"
                if len(s) <= budget:
                    return sign + s + suf
            # Fallback: integer, truncated
            s = str(int(scaled))[:budget]
            return sign + s + suf
    # < 1000: plain integer
    s = f"{a:.0f}"
    if len(sign + s) <= MAX_CHARS:
        return sign + s
    return sign + s[:MAX_CHARS - len(sign)]

def fmt_small(v):
    """For values in range 0-90 (embedding scale)."""
    if abs(v) < 0.001:
        return "0"
    if abs(v - round(v)) < 0.001:
        s = str(int(round(v)))
    else:
        s = f"{v:.3f}"
    return s[:MAX_CHARS]

def fmt_decimal(v):
    """For small decimal values like attention outputs."""
    if abs(v) < 1e-6:
        return "0"
    s = f"{v:.4f}"
    return s[:MAX_CHARS]

def fmt_large(v):
    return _fmt_suffixed(v)

def fmt_sci(v):
    return _fmt_suffixed(v)


# ── Color mapping ─────────────────────────────────────────────────────────
import math

def _symlog(x, linthresh=1.0):
    """Symmetric log: linear in [-linthresh, linthresh], log outside."""
    if abs(x) <= linthresh:
        return x / linthresh
    return math.copysign(1.0 + math.log10(abs(x) / linthresh), x)

def value_to_color(val, vmin, vmax):
    """Diverging blue-white-red if signed, sequential dark-to-bright otherwise.

    Uses symmetric log scaling so huge dynamic ranges are visible.
    """
    if vmax == vmin:
        return rl.Color(128, 128, 128, 255)

    # Choose a linear threshold: values smaller than this are mapped linearly.
    # Use 1% of the larger magnitude, minimum 1.0.
    mag = max(abs(vmin), abs(vmax))
    linthresh = max(1.0, mag * 0.01)

    sv = _symlog(val, linthresh)
    smin = _symlog(vmin, linthresh)
    smax = _symlog(vmax, linthresh)

    if smax == smin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (sv - smin) / (smax - smin)))

    if vmin < 0 and vmax > 0:
        # Diverging: blue(0) -> white(0.5) -> red(1.0)
        if t < 0.5:
            s = t * 2
            r = int(30 + 225 * s)
            g = int(60 + 195 * s)
            b = int(200 + 55 * s)
        else:
            s = (t - 0.5) * 2
            r = 255
            g = int(255 - 200 * s)
            b = int(255 - 230 * s)
    else:
        # Sequential: dark purple -> bright yellow
        r = int(20 + 235 * t)
        g = int(10 + 220 * t * t)
        b = int(80 - 50 * t)
    return rl.Color(r, g, b, 255)


def text_color_for_bg(bg):
    """Return black or white, whichever contrasts more with the given background."""
    # Relative luminance (sRGB linearized)
    def srgb(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * srgb(bg.r) + 0.7152 * srgb(bg.g) + 0.0722 * srgb(bg.b)
    return rl.BLACK if lum > 0.179 else rl.WHITE


# ── Text drawing helpers ─────────────────────────────────────────────────
def draw_mono(text, x, y, size, color):
    """Draw text in the monospace font (ProFontIIx). Use for numbers and data."""
    rl.draw_text_ex(font_mono, text, rl.Vector2(x, y), size, 1, color)

def draw_text(text, x, y, size, color):
    """Draw text in the proportional font (NotoSans). Use for labels and prose."""
    rl.draw_text_ex(font_text, text, rl.Vector2(x, y), size, 1, color)


# ── TensorBlock ───────────────────────────────────────────────────────────
@dataclass
class TensorBlock:
    name: str
    key: str
    data: list              # list of lists [seq_len][dims] (original orientation)
    x: float
    y: float
    seq_len: int            # number of token positions (drawn as columns)
    dims: int               # number of dimensions (drawn as rows)
    dim_labels: list        # dimension names (row labels on the left)
    annotations: list       # list of strings (description notes)
    math_notes: list        # list of strings (MATH: computation details)
    color_range: tuple      # (vmin, vmax)
    format_fn: object       # callable
    argmin_col: int = -1    # if >= 0, highlight argmin cell in this column


# ── Tensor extraction ─────────────────────────────────────────────────────
def run_and_capture(model, a, b, step_index):
    """Run forward pass for one autoregressive step, return all intermediates."""
    correct_sum = a + b
    S = f"{a:010d}+{b:010d}="
    sum_str = f"{correct_sum:011d}"
    for i in range(step_index):
        S += sum_str[i]

    toks = [TOKENS.index(t) for t in ["<bos>"] + list(S)]
    token_labels = ["<bos>"] + list(S)
    x = torch.tensor(toks).unsqueeze(0)

    out = {"token_labels": token_labels}

    def save(t):
        """Convert a 2D tensor [S, D] to list-of-lists."""
        return t.tolist()

    with torch.inference_mode():
        h = model.embedding(x)
        h = pad_to(h, EMBEDDING_DIM)
        out["embedding"] = save(h[0])

        attn0 = model.layer0_attn(h)
        out["l0_attn_out"] = save(attn0[0])
        h = h + attn0
        out["l0_attn_residual"] = save(h[0])

        ffn0 = model.layer0_ffn(h)
        out["l0_ffn_out"] = save(ffn0[0])
        h = pad_to(h, LAYER1_D_MODEL)
        h[..., CANDIDATES_START:LAYER1_D_MODEL] = (
            h[..., CANDIDATES_START:LAYER1_D_MODEL] + ffn0
        )
        out["l0_ffn_residual"] = save(h[0])

        attn1 = model.layer1_attn(h)
        out["l1_attn_out"] = save(attn1[0])
        h = h + attn1
        out["l1_attn_residual"] = save(h[0])

        candidates = h[..., CANDIDATES_START:CANDIDATES_START + NUM_DIGITS]
        out["candidates_pre_ffn"] = save(candidates[0])
        ffn1 = model.layer1_ffn(candidates)
        out["l1_ffn_out"] = save(ffn1[0])
        h = pad_to(h, NUM_DIGITS)
        h = h + ffn1
        out["final_argmin_input"] = save(h[0])

        result = h.argmin(dim=-1)
        out["result"] = result[0].tolist()

    return out


# ── Layout ────────────────────────────────────────────────────────────────
BLOCK_DEFS = [
    # (name, key, dim_labels, color_range, fmt, annotations, math_notes)
    ("Embedding", "embedding",
     ["EQ", "SPECIAL", "DIGIT", "COUNT", "SCALE"],
     (0, 90), fmt_small,
     ["Sparse lookup: 14 tokens -> 5 dims, 13 non-zero values.",
      "Digits: value * 10 in DIGIT dim. Token '0' -> all zeros.",
      "Special tokens (<bos>, +, =) set SPECIAL=1. '=' also sets EQ=1.",
      "COUNT and SCALE dims left at 0 -- filled by attention."],
     ["EQ[t]    = 1 if t == '='  else 0",
      "SPECIAL[t] = 1 if t in {<bos>,+,=} else 0",
      "DIGIT[t] = int(t) * 10  (DIGIT_EMBED_SCALE=10)",
      "COUNT[t] = 0  (reserved for L0 attention)",
      "SCALE[t] = 0  (reserved for L0 attention)"]),

    ("Layer 0 Attention Output", "l0_attn_out",
     ["EQ", "SPECIAL", "DIGIT", "COUNT", "SCALE"],
     None, fmt_decimal,
     ["5 heads (d_k=1), but only heads 3 & 4 are active.",
      "Head 3 -> COUNT: ALiBi slope=log(10) creates 10^(-distance) counter.",
      "  Resets at each special token. Negative after '='.",
      "Head 4 -> SCALE: gate signal ~1/(pos+2), non-zero only after '='.",
      "Uses softmax1 (can 'attend to nothing')."],
     ["Head 3 (COUNT dim):",
      "  K = SPECIAL * 960 - 1000  (special:-40, digit:-1000)",
      "  V = (SPECIAL*0.1 + EQ*-1.1) / V_PROJ_SCALE",
      "  ALiBi slope = log(10) ~ 2.3026",
      "  attn = softmax1(Q*K + ALiBi + causal_mask)",
      "  Result: 10^(-dist_from_special), neg after '='",
      "",
      "Head 4 (SCALE dim):",
      "  V = EQ (1 for '=', 0 otherwise)",
      "  Q=K=0, softmax1 -> uniform: 1/(pos+2)",
      "  Result: ~0.03-0.04 after '=', 0 before"]),

    ("Layer 0 Attn + Residual", "l0_attn_residual",
     ["EQ", "SPECIAL", "DIGIT", "COUNT", "SCALE"],
     None, fmt_small,
     ["Embedding (dims 0-2) + Attention (dims 3-4) combine cleanly.",
      "All 5 channels now populated."],
     ["h = embedding + attn_output",
      "No overlap: embed uses dims 0-2, attn uses dims 3-4."]),

    ("Layer 0 FFN Output", "l0_ffn_out",
     ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "place"],
     None, fmt_large,
     ["Gated FFN: output = ReLU(gate) * up. Widens 5 -> 11 dims.",
      "Before '=': slot 10 = digit * 10^place (place-value contributions).",
      "After '=': slots 0-9 = candidate scores (large negative ramp)."],
     ["output = ReLU(gate) * up",
      "",
      "Slots 0-9 (candidate scores, after '=' only):",
      "  gate[d] = SCALE    (>0 only after '=')",
      "  up[d]   = COUNT * (d+0.5) * 1e10 * 100",
      "  Result: -(d+0.5) * SCALE * COUNT * 1e12",
      "",
      "Slot 10 (place value, before '=' only):",
      "  gate[10] = DIGIT   (digit_val * 10)",
      "  up[10]   = COUNT * 1e10",
      "  Result: digit * 10 * 10^(-dist) * 1e10"]),

    ("Layer 0 FFN + Residual [16 dims]", "l0_ffn_residual",
     ["EQ", "SP", "DIG", "CNT", "SCL",
      "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "pv"],
     None, fmt_large,
     ["Residual widened 5 -> 16. FFN output added to dims 5-15.",
      "Dims 0-4: original embedding + attention values.",
      "Dims 5-14: candidate digit scores (only active after '=').",
      "Dim 15: place-value encoded digit contribution."],
     ["h = pad_to(h, 16)",
      "h[5:16] += ffn_output[0:11]",
      "Dims 0-4 pass through unchanged."]),

    ("Layer 1 Attention Output", "l1_attn_out",
     ["sum"],
     None, fmt_large,
     ["Q=K=0: uniform causal attention -> average all prior V values.",
      "V = dim15 * 100 + 15. Encodes (a+b)*1000 as running sum.",
      "Previously generated answer digits subtract their place values.",
      "This is like long addition with a running remainder."],
     ["V[pos] = h[pos, dim15] * FINAL_SCALE + GATE_BIAS_SHIFT",
      "       = h[pos, 15] * 100 + 15",
      "",
      "Q=K=0 -> softmax1 -> weight = 1/(pos+2) each",
      "output[pos] = sum(V[0..pos]) / (pos+2)",
      "",
      "Digit V: place_val * 100 + 15",
      "Non-digit V: 0 * 100 + 15 = 15",
      "Answer digit V: -place_val * 100 + 15 (subtracts!)"]),

    ("Layer 1 Attn + Residual", "l1_attn_residual",
     ["EQ", "SP", "DIG", "CNT", "SCL",
      "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "pv"],
     None, fmt_large,
     ["Single sum value broadcast-added to all 16 dims.",
      "Dims 5-14: candidates shifted so correct digit is nearest 0."],
     ["h = h + attn_output  (broadcast [S,1] -> [S,16])",
      "All dims shifted by same amount.",
      "Candidates (dims 5-14) = old_score + running_sum.",
      "Correct digit lands closest to 0."]),

    ("Candidates (pre-FFN)", "candidates_pre_ffn",
     ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"],
     None, fmt_large,
     ["Extracted dims 5-14 from residual stream.",
      "The correct answer digit has value closest to 0."],
     ["candidates = h[:, 5:15]",
      "Just a slice -- no computation."]),

    ("Layer 1 FFN Output (|x| * 1M)", "l1_ffn_out",
     ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"],
     None, fmt_sci,
     ["|x| via ReLU(x*1e4) + ReLU(-x*1e4), then * 100.",
      "'Closest to zero' becomes 'smallest value'.",
      "This is the V-shaped absolute value trick via dual ReLU."],
     ["output = (ReLU(x * V_SCALE) + ReLU(x * -V_SCALE)) * FINAL_SCALE",
      "       = |x| * V_SCALE * FINAL_SCALE",
      "       = |x| * 1e4 * 100  =  |x| * 1e6",
      "",
      "V_SCALE = 1e4,  FINAL_SCALE = 100"]),

    ("Final Argmin Input", "final_argmin_input",
     ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"],
     None, fmt_sci,
     ["argmin(dim=-1) selects the predicted digit.",
      "Residual is negligible vs FFN output (billions).",
      "The minimum cell in the last column is the prediction."],
     ["h = pad_to(h, 10) + ffn1_output",
      "predicted_digit = argmin(h[-1])",
      "",
      "Residual (thousands) << FFN (billions),",
      "so argmin is determined entirely by FFN."]),
]


def build_layout(tensors):
    """Position all tensor blocks top-to-bottom.

    Each block is transposed for display: token positions are columns,
    dimensions are rows. Blocks are left-aligned at x=0.
    """
    blocks = []
    y = 0.0

    for name, key, dim_labels, color_range, fmt, annotations, math_notes in BLOCK_DEFS:
        data = tensors[key]
        # Ensure 2D list-of-lists [seq_len][dims]
        if isinstance(data[0], (int, float)):
            data = [[v] for v in data]
        seq_len = len(data)
        dims = len(data[0])

        if color_range is None:
            flat = [v for row in data for v in row]
            vmin, vmax = min(flat), max(flat)
            if vmin < 0 < vmax:
                m = max(abs(vmin), abs(vmax))
                vmin, vmax = -m, m
            elif vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            color_range = (vmin, vmax)

        argmin_col = -1
        if key == "final_argmin_input":
            argmin_col = seq_len - 1

        blocks.append(TensorBlock(
            name=name, key=key, data=data,
            x=0, y=y, seq_len=seq_len, dims=dims,
            dim_labels=dim_labels,
            annotations=annotations,
            math_notes=math_notes,
            color_range=color_range,
            format_fn=fmt,
            argmin_col=argmin_col,
        ))
        y += dims * CELL_SIZE + BLOCK_GAP

    return blocks


# ── Drawing ───────────────────────────────────────────────────────────────
def draw_flow_arrow(y_start, y_end, x_center):
    """Downward arrow between blocks."""
    mid = int(x_center)
    rl.draw_line(mid, int(y_start), mid, int(y_end - 8), ARROW_COLOR)
    rl.draw_triangle(
        rl.Vector2(x_center - 6, y_end - 10),
        rl.Vector2(x_center + 6, y_end - 10),
        rl.Vector2(x_center, y_end),
        ARROW_COLOR,
    )


def draw_tensor_block(block, camera, token_labels):
    """Draw one tensor block (transposed: columns=tokens, rows=dims)."""
    zoom = camera.zoom
    screen_cell = CELL_SIZE * zoom
    vmin, vmax = block.color_range

    # Block title (always visible)
    draw_text(block.name, block.x, block.y - 28, 18, LABEL_COLOR)

    # Level 2+: column headers (token labels along the top)
    if screen_cell >= LEVEL2_THRESH:
        hdr_fs = max(6, CELL_SIZE * 0.3)
        for c in range(block.seq_len):
            if c < len(token_labels):
                cx = block.x + c * CELL_SIZE + 2
                draw_mono(token_labels[c], cx, block.y - 14, hdr_fs, DIM_LABEL_COLOR)
        # Shape
        shape_txt = f"[{block.seq_len}x{block.dims}]"
        draw_text(shape_txt, block.x + block.seq_len * CELL_SIZE + 8,
                  block.y, 12, rl.GRAY)

    # Level 3+: row labels (dimension names on the left)
    if screen_cell >= LEVEL3_THRESH:
        lbl_fs = max(6, CELL_SIZE * 0.3)
        for r in range(block.dims):
            if r < len(block.dim_labels):
                cy = block.y + r * CELL_SIZE + CELL_SIZE * 0.3
                draw_mono(block.dim_labels[r],
                          block.x - ROW_LABEL_W, cy, lbl_fs, ROW_LABEL_COLOR)

    # Gray background behind the grid (visible as cell borders/gaps)
    rl.draw_rectangle(int(block.x - 1), int(block.y - 1),
                      int(block.seq_len * CELL_SIZE + 1),
                      int(block.dims * CELL_SIZE + 1), rl.GRAY)

    # Heatmap cells (transposed: col=token position, row=dimension)
    val_fs = max(5, CELL_SIZE * 0.25)
    for pos in range(block.seq_len):
        for dim in range(block.dims):
            v = float(block.data[pos][dim])
            color = value_to_color(v, vmin, vmax)
            cx = block.x + pos * CELL_SIZE
            cy = block.y + dim * CELL_SIZE
            rl.draw_rectangle(int(cx), int(cy),
                              int(CELL_SIZE - 1), int(CELL_SIZE - 1), color)

            # Level 3+: numeric values and borders
            if screen_cell >= LEVEL3_THRESH:
                txt = block.format_fn(v)
                draw_mono(txt, cx + 2, cy + CELL_SIZE * 0.35,
                          val_fs, text_color_for_bg(color))
                rl.draw_rectangle_lines(int(cx-1), int(cy-1),
                                int(CELL_SIZE + 1), int(CELL_SIZE + 1),
                                rl.GRAY)
            

    # Argmin highlight (last token column)
    if block.argmin_col >= 0:
        col = block.argmin_col
        col_vals = [block.data[col][d] for d in range(block.dims)]
        min_d = min(range(len(col_vals)), key=lambda i: col_vals[i])
        hx = block.x + col * CELL_SIZE
        hy = block.y + min_d * CELL_SIZE
        rl.draw_rectangle_lines(int(hx - 1), int(hy - 1),
                                int(CELL_SIZE + 1), int(CELL_SIZE + 1),
                                HIGHLIGHT_COLOR)
        rl.draw_rectangle_lines(int(hx - 2), int(hy - 2),
                                int(CELL_SIZE + 3), int(CELL_SIZE + 3),
                                HIGHLIGHT_COLOR)

    # Level 4: annotations (below the heatmap, floating horizontally)
    if screen_cell >= LEVEL4_THRESH:
        ann_fs = 12
        ann_col_w = 420  # world-space width of each annotation column
        ann_gap = 20     # gap between the two columns
        block_left = block.x
        block_right = block.x + block.seq_len * CELL_SIZE
        max_float_right = block_right - 700  # don't float too far right

        # Convert screen left edge to world x to determine float position
        screen_left_world = rl.get_screen_to_world_2d(rl.Vector2(20, 0), camera).x
        # Float: track screen left edge, but clamp within limits
        ann_x = max(block_left, min(screen_left_world, max_float_right))

        ann_y = block.y + block.dims * CELL_SIZE + 8

        # Left column: description notes
        draw_text("NOTES:", ann_x, ann_y, ann_fs, rl.Color(200, 200, 210, 255))
        ny = ann_y + 16
        for line in block.annotations:
            draw_text(line, ann_x, ny, ann_fs, ANNOTATION_COLOR)
            ny += 16

        # Right column: math notes
        math_x = ann_x + ann_col_w + ann_gap
        draw_mono("MATH:", math_x, ann_y, ann_fs, rl.Color(200, 200, 210, 255))
        my = ann_y + 16
        for line in block.math_notes:
            draw_mono(line, math_x, my, ann_fs, ANNOTATION_COLOR)
            my += 16


# ── Input handling ────────────────────────────────────────────────────────
def handle_camera(camera):
    """Zoom toward cursor with mouse wheel, pan with right-drag."""
    wheel = rl.get_mouse_wheel_move()
    if wheel != 0:
        mouse_before = rl.get_screen_to_world_2d(rl.get_mouse_position(), camera)
        camera.zoom *= 1.1 ** wheel
        camera.zoom = max(0.02, min(camera.zoom, 15.0))
        mouse_after = rl.get_screen_to_world_2d(rl.get_mouse_position(), camera)
        camera.target.x += mouse_before.x - mouse_after.x
        camera.target.y += mouse_before.y - mouse_after.y

    if (rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) or
            rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)):
        delta = rl.get_mouse_delta()
        camera.target.x -= delta.x / camera.zoom
        camera.target.y -= delta.y / camera.zoom


def draw_text_input(text_buf, active):
    """Draw input text field in screen space."""
    bx, by, bw, bh = 20, 10, 320, 30
    border = rl.YELLOW if active else rl.GRAY
    rl.draw_rectangle(bx, by, bw, bh, rl.Color(50, 50, 55, 255))
    rl.draw_rectangle_lines(bx, by, bw, bh, border)
    draw_mono(text_buf, bx + 8, by + 7, 18, rl.WHITE)
    if not active:
        draw_text("(click or Tab to edit)", bx + bw + 10, by + 9, 14, rl.GRAY)


def handle_text_input(text_buf, active):
    """Returns (text_buf, active, recompute)."""
    recompute = False

    if rl.is_key_pressed(rl.KeyboardKey.KEY_TAB):
        active = not active
    if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
        mx, my = rl.get_mouse_x(), rl.get_mouse_y()
        active = 20 <= mx <= 340 and 10 <= my <= 40

    if not active:
        return text_buf, active, recompute

    ch = rl.get_char_pressed()
    while ch != 0:
        c = chr(ch)
        if c in "0123456789+":
            text_buf += c
        ch = rl.get_char_pressed()

    if rl.is_key_pressed(rl.KeyboardKey.KEY_BACKSPACE) and text_buf:
        text_buf = text_buf[:-1]

    if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
        active = False
        recompute = True

    return text_buf, active, recompute


def parse_input(text):
    """Parse 'A+B' from text. Returns (a, b) or None on failure."""
    try:
        parts = text.split("+")
        a, b = int(parts[0]), int(parts[1])
        max_val = 10 ** NUM_DIGITS - 1
        a = max(0, min(a, max_val))
        b = max(0, min(b, max_val))
        if a + b > max_val:
            b = max_val - a
        return a, b
    except (ValueError, IndexError):
        return None


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global font_mono, font_text
    rl.init_window(SCREEN_W, SCREEN_H, "TinyAdder Visualizer")
    rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.set_target_fps(60)

    font_mono = rl.load_font_ex(MONO_FONT_PATH, FONT_SIZE, None, 0)
    font_text = rl.load_font_ex(TEXT_FONT_PATH, FONT_SIZE, None, 0)
    rl.set_texture_filter(font_mono.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_filter(font_text.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    camera = rl.Camera2D(
        rl.Vector2(SCREEN_W / 2, 50),   # offset
        rl.Vector2(0, 0),               # target
        0.0,                             # rotation
        0.3,                             # zoom
    )

    model = TinyAdderModule()
    input_a, input_b = 1234, 5678
    step_index = 8  # predicting the 9th digit (index 8)

    tensors = run_and_capture(model, input_a, input_b, step_index)
    blocks = build_layout(tensors)
    token_labels = tensors["token_labels"]
    predicted = int(tensors["result"][-1])

    text_buf = f"{input_a}+{input_b}"
    text_active = False

    while not rl.window_should_close():
        # ── Input ──
        handle_camera(camera)

        # Step selector (left/right arrows, only when text input not active)
        step_changed = False
        if not text_active:
            if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT) and step_index > 0:
                step_index -= 1
                step_changed = True
            if rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT) and step_index < NUM_DIGITS:
                step_index += 1
                step_changed = True

        text_buf, text_active, recompute = handle_text_input(text_buf, text_active)

        if recompute:
            parsed = parse_input(text_buf)
            if parsed:
                input_a, input_b = parsed
                text_buf = f"{input_a}+{input_b}"

        if recompute or step_changed:
            tensors = run_and_capture(model, input_a, input_b, step_index)
            blocks = build_layout(tensors)
            token_labels = tensors["token_labels"]
            predicted = int(tensors["result"][-1])

        # ── Draw ──
        rl.begin_drawing()
        rl.clear_background(BG_COLOR)

        # World-space content
        rl.begin_mode_2d(camera)

        # Flow arrows between blocks (centered on block width)
        for i in range(len(blocks) - 1):
            b0 = blocks[i]
            b1 = blocks[i + 1]
            arrow_x = b0.x + b0.seq_len * CELL_SIZE / 2
            y_start = b0.y + b0.dims * CELL_SIZE + 5
            y_end = b1.y - 5
            draw_flow_arrow(y_start, y_end, arrow_x)

        for block in blocks:
            draw_tensor_block(block, camera, token_labels)

        rl.end_mode_2d()

        # Screen-space UI
        rl.draw_rectangle(0, 0, 500, 128, rl.Color(0, 0, 0, 128))
        draw_text_input(text_buf, text_active)

        # Step indicator and result
        correct_sum = input_a + input_b
        sum_str = f"{correct_sum:011d}"
        info = f"Step {step_index}/10  |  {input_a} + {input_b} = {correct_sum}"
        draw_text(info, 20, 50, 18, rl.WHITE)

        generated_so_far = sum_str[:step_index]
        predicting = f"Generated: {generated_so_far}_  |  Next digit: {predicted}"
        draw_mono(predicting, 20, 75, 16, rl.Color(255, 255, 150, 255))

        ctx_display = "".join(token_labels)
        draw_mono(f"Context: {ctx_display}", 20, 100, 14, rl.Color(150, 150, 160, 255))

        zoom_pct = f"Zoom: {camera.zoom:.2f}x"
        draw_text(zoom_pct, rl.get_screen_width() - 140, 10, 14, rl.GRAY)

        controls = "Mouse wheel: zoom | Mouse drag: pan | Left/Right arrow: step | Tab: edit input"
        draw_text(controls, 20, rl.get_screen_height() - 25, 12, rl.Color(100, 100, 110, 255))

        rl.end_drawing()

    rl.unload_font(font_mono)
    rl.unload_font(font_text)
    rl.close_window()


if __name__ == "__main__":
    main()
