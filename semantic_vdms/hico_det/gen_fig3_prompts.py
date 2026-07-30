#!/usr/bin/env python3
"""
Generate corrected Figure 3: LLM Prompt Structure.
Fixes all Critical discrepancies from Apr-27-2026 audit.
Box heights are computed from actual text content — no dead space.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colours ──────────────────────────────────────────────────────────
CQ  = '#E3F2FD'
CHD = '#DDEAF7'
CPH = '#E8F5E9'
CGU = '#FFF3E0'
CHI = '#F5F5F5'
CHT = '#EDE7F6'
CJS = '#F3E5F5'
COK = '#1B5E20'
CFR = '#B71C1C'
CDB = '#1A237E'
CDP = '#4A148C'
CDG = '#222222'
CGN = '#1B5E20'
CBR = '#4E342E'

# ── Figure & layout constants ─────────────────────────────────────────
FW, FH = 7.4, 8.0   # inches (bbox_inches='tight' crops)
fig = plt.figure(figsize=(FW, FH))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')

# Coordinate unit = 1 inch.  fontsize N pt ≈ N/72 inch tall.
# Line height for 6.6–7.5pt text: use LH = 0.145 in (≈ 10.4pt leading).
LH   = 0.145
TP   = 0.11   # top padding inside a section box
BP   = 0.07   # bottom padding inside a section box
MRG  = 0.16
BX   = MRG
BW   = FW - 2*MRG
IPX  = BX + 0.12
IPW  = BW - 0.24
TXT  = IPX + 0.08   # text left edge
GAP  = 0.050  # gap between inner sections


def rbox(x, y, w, h, fc, ec='#999', lw=0.65, r=0.07, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False))


def arr(x0, y0, x1, y1, lw=1.1, color='#555'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=11))


def T(x, y, s, **kw):
    kw.setdefault('ha', 'left'); kw.setdefault('va', 'top')
    kw.setdefault('zorder', 4); kw.setdefault('color', CDG)
    ax.text(x, y, s, **kw)


# ════════════════════════════════════════════════════════════════════
# Section height = TP + (n_lines – 1)*LH_spacing + text_row_height + BP
# text_row_height ≈ 0.095 in for 6.6–7.5 pt fonts
# LH_spacing between consecutive text rows as coded below.
# ════════════════════════════════════════════════════════════════════
TR = 0.095   # single text-row height

# Pre-compute heights from content
#   Each entry: list of inter-line gaps BEFORE each line (0 for first line after TP).
#   Height = TP + sum(gaps) + TR + BP

def sec_h(gaps):
    """gaps = list of LH multipliers between consecutive lines (len = n_lines - 1)."""
    return TP + sum(g * LH for g in gaps) + TR + BP


# Section heights (computed from actual gaps in text below)
HD_H = sec_h([1.0, 1.0])           # 3 lines: title, iter, objective
PH_H = sec_h([1.0, 0.90, 0.90])    # 4 lines: title, k-line, efS-line, IVF-line
GU_H = sec_h([1.0, 1.0])           # 3 items: title+class (one row), bracket, directive
HI_H = sec_h([1.05, 0.82, 1.05, 0.82, 1.05, 0.82])  # title + 3×(label+result)
HT_H = sec_h([1.0, 1.0])           # 3 lines: title, active note, dedup note
JS_H = sec_h([1.0, 0.90, 0.90, 0.90, 0.90])  # title + 5 JSON lines


# ════════════════════════════════════════════════════════════════════
# Layout BOTTOM → UP
# ════════════════════════════════════════════════════════════════════
cy = 0.14   # y of JSON box bottom

# ── JSON RESPONSE ────────────────────────────────────────────────────
rbox(BX, cy, BW, JS_H, CJS, ec='#8E24AA', lw=0.9, r=0.10, z=1)
hy = cy + JS_H - TP
T(IPX, hy, "Structured JSON Response — Iteration 4:",
  fontsize=7.5, fontweight='bold', color=CDP)
hy -= LH * 1.0
T(IPX, hy, '{ "engine": "FaissHNSWFlat",',
  fontsize=6.6, fontfamily='monospace', color=CDB)
hy -= LH * 0.90
T(IPX, hy, '  "params": {"M": 64, "efConstruction": 200, "efSearch": 64},',
  fontsize=6.6, fontfamily='monospace', color=CDB)
hy -= LH * 0.90
T(IPX, hy, '  "k_neighbors": 50,  "alpha": 0.70,  "n_refs": 10,  "ref_strategy": "centroid",',
  fontsize=6.6, fontfamily='monospace', color=CDB)
hy -= LH * 0.90
T(IPX, hy, '  "constraint_strategy": "object",       ← new field (PMGD graph pre-filter)',
  fontsize=6.6, fontfamily='monospace', color='#7B1FA2', fontweight='bold')
hy -= LH * 0.90
T(IPX, hy, '  "reasoning": "k=50→QPS≈295; M=64+cs=object keeps mAP≥τ=0.15." }',
  fontsize=6.6, fontfamily='monospace', color=CDB)

json_top = cy + JS_H

# ── ARROW: main box → JSON ───────────────────────────────────────────
A1_BOT = json_top + 0.02
A1_TOP = json_top + 0.28
arr(FW/2, A1_TOP, FW/2, A1_BOT)
T(FW/2 + 0.10, (A1_BOT + A1_TOP)/2,
  "MiniMax-M2.1 via OpenRouter",
  ha='left', va='center', fontsize=7, color='#444', style='italic')

# ── INNER SECTIONS (bottom → top) ────────────────────────────────────
sec_bot = A1_TOP + 0.07
iy = sec_bot

# Section 5 — HINTS ──────────────────────────────────────────────────
rbox(IPX, iy, IPW, HT_H, CHT, r=0.06, z=3)
hy = iy + HT_H - TP
T(TXT, hy, "Untried-value hints  Uₜ :",
  fontsize=7.2, fontweight='bold', color=CDP)
hy -= LH * 1.0
T(TXT, hy, "Active during Exploitation & Fine-Tuning only (suppressed in Exploration phase).",
  fontsize=6.8, color=CDP, style='italic')
hy -= LH * 1.0
T(TXT, hy, "Each θₙ₊₁ checked against ℋₜ; duplicates trigger resubmission or random fallback.",
  fontsize=6.8, color='#555', style='italic')
iy += HT_H + GAP

# Section 4 — HISTORY ────────────────────────────────────────────────
rbox(IPX, iy, IPW, HI_H, CHI, r=0.06, z=3)
hy = iy + HI_H - TP
T(TXT, hy, "History  ℋ₃  (complete ordered trajectory — no truncation):",
  fontsize=7.2, fontweight='bold')
hy -= LH * 1.05
T(TXT, hy, "iter01:  HNSW  M=64  efS=64  k=100  α=0.65  cs=object  refs=10+centroid",
  fontsize=6.6, fontfamily='monospace')
hy -= LH * 0.82
T(TXT, hy, "         →  Score=136.1   mAP=0.209   QPS=136.1   ✓",
  fontsize=6.6, fontfamily='monospace', color=COK, fontweight='bold')
hy -= LH * 1.05
T(TXT, hy, "iter02:  IVFFlat  nlist=128  nprobe=8   k=100  α=0.65  cs=none   refs=10+centroid",
  fontsize=6.6, fontfamily='monospace')
hy -= LH * 0.82
T(TXT, hy, "         →  Score=  0.0   mAP=0.000   QPS=2267.6   ✗  (mAP < τ — infeasible)",
  fontsize=6.6, fontfamily='monospace', color=CFR, fontweight='bold')
hy -= LH * 1.05
T(TXT, hy, "iter03:  IVFFlat  nlist=512  nprobe=64  k=100  α=0.30  cs=none   refs=5+first",
  fontsize=6.6, fontfamily='monospace')
hy -= LH * 0.82
T(TXT, hy, "         →  Score=  0.0   mAP=0.000   QPS=2281.6   ✗  (mAP < τ — infeasible)",
  fontsize=6.6, fontfamily='monospace', color=CFR, fontweight='bold')
iy += HI_H + GAP

# Section 3 — GUIDANCE ───────────────────────────────────────────────
rbox(IPX, iy, IPW, GU_H, CGU, r=0.06, z=3)
hy = iy + GU_H - TP
T(TXT, hy, "Diagnostic  g(ℋ₃) :",
  fontsize=7.2, fontweight='bold', color='#E65100')
T(TXT + 1.32, hy, "Class = low_mAP",
  fontsize=7.2, fontweight='bold', fontfamily='monospace', color=CFR)
hy -= LH * 1.0
T(TXT, hy,
  "[ mAP(θ₃) = 0.000 < 0.20  and  Score(θ₃) = 0.0 < 0.80 × Score*₃ = 108.9 ]",
  fontsize=6.8, color='#555', style='italic')
hy -= LH * 1.0
T(TXT, hy,
  "⇒ IVFFlat returned mAP=0.000 (infeasible). Return to FaissHNSWFlat. Increase k or M.",
  fontsize=6.8, color=CBR)
iy += GU_H + GAP

# Section 2 — PHYSICS ────────────────────────────────────────────────
rbox(IPX, iy, IPW, PH_H, CPH, r=0.06, z=3)
hy = iy + PH_H - TP
T(TXT, hy, "Parameter Physics  (190-line system context — key constraints summarised):",
  fontsize=7.2, fontweight='bold', color=CGN)
hy -= LH * 1.0
T(TXT, hy, "k_neighbors ↓  ⇒  QPS ↑↑  (primary lever: k=50→295, k=100→145, k=200→58 QPS)",
  fontsize=6.8, fontfamily='monospace')
hy -= LH * 0.90
T(TXT, hy, "efSearch: QPS-neutral in VDMS batch mode  ·  M ↑  ⇒  mAP ceiling ↑  ·  α: discover empirically",
  fontsize=6.8, fontfamily='monospace')
hy -= LH * 0.90
T(TXT, hy, "IVFFlat + constraint_strategy ≠ 'none'  ⇒  0 candidates returned  ⚠  (always infeasible)",
  fontsize=6.8, fontfamily='monospace', color=CFR)
iy += PH_H + GAP

# Section 1 — HEADER (topmost inner) ────────────────────────────────
rbox(IPX, iy, IPW, HD_H, CHD, r=0.06, z=3)
hy = iy + HD_H - TP
T(TXT, hy, "You are an expert VDMS Optimization Engine for HOI Retrieval.",
  fontsize=7.5, fontweight='bold')
hy -= LH * 1.0
T(TXT, hy, "Iteration: 4 / 50     |     Phase: EXPLORATION     |     Best Score so far: 136.1",
  fontsize=7, fontfamily='monospace', color=CDB)
hy -= LH * 1.0
T(TXT, hy,
  "OBJECTIVE:  Score = QPS  if  mAP ≥ τ (τ=0.15)  else  0.0   ⇒   maximize QPS s.t. mAP ≥ 0.15",
  fontsize=7, fontfamily='monospace', color=CFR, fontweight='bold')
iy += HD_H

# ── Outer "LLM User Message" box ─────────────────────────────────────
OPAD  = 0.09
O_BOT = sec_bot - OPAD
O_TOP = iy + OPAD + 0.20
O_H   = O_TOP - O_BOT
rbox(BX, O_BOT, BW, O_H, 'white', ec='#444', lw=1.05, r=0.12, z=1)
ax.text(BX + 0.20, O_TOP - 0.01,
        "LLM User Message",
        ha='left', va='bottom', fontsize=8.5, fontweight='bold', color='#222',
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                  edgecolor='#444', lw=0.55), zorder=5)

# ── ARROW: queries → outer box ───────────────────────────────────────
A0_BOT = O_TOP
A0_TOP = O_TOP + 0.22
arr(FW/2, A0_TOP, FW/2, A0_BOT)

# ── HOI QUERY BOXES ───────────────────────────────────────────────────
QW, QH = 1.90, 0.68
QY   = A0_TOP
QGAP = 0.16
total_qw = 3*QW + 2*QGAP
qs_x = (FW - total_qw) / 2

for i, (title, l1, l2) in enumerate([
    ("HOI Query", "Person: holding", "Object: bottle"),
    ("HOI Query", "Person: riding",  "Object: horse"),
    ("HOI Query", "···   × 600",     ""),
]):
    qx = qs_x + i*(QW + QGAP)
    rbox(qx, QY, QW, QH, CQ, ec='#90CAF9', lw=0.65, r=0.08, z=3)
    ax.text(qx+QW/2, QY+QH-0.10, title,
            ha='center', va='top', fontsize=7.5, fontweight='bold', color=CDG, zorder=4)
    ax.text(qx+QW/2, QY+QH/2+0.02, l1,
            ha='center', va='center', fontsize=7, fontfamily='monospace', color=CDG, zorder=4)
    if l2:
        ax.text(qx+QW/2, QY+0.10, l2,
                ha='center', va='center', fontsize=7, fontfamily='monospace', color=CDG, zorder=4)

# ── Save ─────────────────────────────────────────────────────────────
OUT = "/path/to/paper/figures/prompts.pdf"
plt.savefig(OUT, format='pdf', bbox_inches='tight', dpi=200)
print(f"Saved: {OUT}")
print(f"Section heights: HD={HD_H:.3f} PH={PH_H:.3f} GU={GU_H:.3f} HI={HI_H:.3f} HT={HT_H:.3f} JS={JS_H:.3f}")
