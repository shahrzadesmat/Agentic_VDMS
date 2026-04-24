#!/usr/bin/env python3
"""
Generate Figure 2: Agentic VDMS workflow diagram (VLDB submission).
All mandatory fixes applied:
  - SIEVE metric (QPS if metric >= tau else 0)
  - FaissHNSWFlat in Config Proposal + VDMS (highlighted)
  - 50-iter phase bounds: 1-20 / 21-38 / 39-50
  - GP-BO as 5th method
  - GLDv2 as 3rd benchmark
  - N_q (generalised query count)
  - History Buffer H_t on LLM path only
  - "50 iterations" in Best Configuration box
Output: workflow.pdf in same directory.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.path import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'workflow.pdf')

# ── Figure canvas ────────────────────────────────────────────────────────────
FW, FH = 20.0, 9.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette ──────────────────────────────────────────────────────────────────
BLU,  BLU_E  = '#dae8fc', '#6c8ebf'
YEL,  YEL_E  = '#fff2cc', '#d6b656'
GRN,  GRN_E  = '#d5e8d4', '#82b366'
ORN,  ORN_E  = '#ffe6cc', '#d79b00'
GRY,  GRY_E  = '#f5f5f5', '#999999'
HNSW, HNSW_E = '#c7f0c7', '#1d7a1d'
HIST, HIST_E = '#fce4d6', '#c0392b'
REDD          = '#b85450'
VBKG, VBKG_E = '#d5e8d4', '#2d6a2d'

# ── Primitives ────────────────────────────────────────────────────────────────
def R(x, y, w, h, fc=GRY, ec=GRY_E, lw=1.5, rr=0.12, z=3, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={rr}',
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, clip_on=False)
    ax.add_patch(p)
    return p

def T(x, y, s, fs=9, fw='normal', ha='center', va='center',
      c='black', z=7, st='normal'):
    return ax.text(x, y, s, fontsize=fs, fontweight=fw, fontstyle=st,
                   ha=ha, va=va, color=c, zorder=z)

def seg(pts, ec='#555', lw=1.5, dashed=False, z=5):
    """Draw a polyline through pts=[(x,y), ...]."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.plot(xs, ys, color=ec, lw=lw, linestyle=ls, zorder=z,
            solid_capstyle='round', dash_capstyle='round')

def arrowhead(x, y, dx, dy, ec, lw, z=5, scale=12):
    """Place an arrowhead at (x,y) pointing in direction (dx,dy)."""
    ax.annotate('', xy=(x, y), xytext=(x - dx*0.001, y - dy*0.001),
                arrowprops=dict(arrowstyle='->', color=ec, lw=lw,
                                mutation_scale=scale),
                zorder=z)

def arrow(x1, y1, x2, y2, ec='#555', lw=1.5, dashed=False,
          lbl=None, lfs=8, lbl_dx=0, lbl_dy=0.15, z=5):
    seg([(x1, y1), (x2, y2)], ec=ec, lw=lw, dashed=dashed, z=z)
    arrowhead(x2, y2, x2-x1, y2-y1, ec, lw, z=z)
    if lbl:
        mx, my = (x1+x2)/2 + lbl_dx, (y1+y2)/2 + lbl_dy
        T(mx, my, lbl, fs=lfs, c=ec, z=z+1)

def routed_arrow(pts, ec='#555', lw=1.5, dashed=False,
                 lbl=None, lfs=8, lbl_idx=None, lbl_dx=0, lbl_dy=0.15, z=5):
    """Multi-segment arrow; arrowhead at last point pointing from second-to-last."""
    seg(pts, ec=ec, lw=lw, dashed=dashed, z=z)
    x2, y2 = pts[-1]; x1, y1 = pts[-2]
    arrowhead(x2, y2, x2-x1, y2-y1, ec, lw, z=z)
    if lbl:
        idx = lbl_idx if lbl_idx is not None else len(pts)//2
        bx = (pts[idx-1][0]+pts[idx][0])/2 + lbl_dx
        by = (pts[idx-1][1]+pts[idx][1])/2 + lbl_dy
        T(bx, by, lbl, fs=lfs, c=ec, z=z+1)

def diamond(cx, cy, w, h, fc=ORN, ec=ORN_E, lw=2.0, z=3):
    pts = [[cx, cy+h/2], [cx+w/2, cy], [cx, cy-h/2], [cx-w/2, cy]]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z,
                         clip_on=False))

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT  (all in inches on 20 × 9.5 canvas)
#   Benchmarks: x = [0.10, 3.10]   w = 3.00
#   Loop:       x = [3.30, 14.50]  w = 11.20
#   VDMS:       x = [14.70, 19.90] w = 5.20
#   All y:      [0.15, 9.30]       h = 9.15
# ════════════════════════════════════════════════════════════════════════════
PAD   = 0.15
TOP   = FH - 0.20
BOT   = 0.15

BX, BW = 0.10, 3.00   # Benchmarks
LX, LW = 3.30, 11.20  # Loop
VX, VW = 14.70, 5.20  # VDMS

H = TOP - BOT  # 9.15

# ════════════════════════════════════════════════════════════════════════════
# 1. BENCHMARKS PANEL
# ════════════════════════════════════════════════════════════════════════════
R(BX, BOT, BW, H, fc=BLU, ec=BLU_E, lw=2.0, rr=0.15, z=1)
T(BX+BW/2, TOP-0.35, 'Benchmarks', fs=13, fw='bold')

# heights for 3 benchmark boxes (leave title + gaps)
INNER_H = H - 0.75
BH_each = (INNER_H - 0.30) / 3     # ~2.85 each
BH_gap  = 0.15
SIEVE_H = 0.42                     # SIEVE formula row height

def bench_box(label, sub_lines, tau_line, by):
    bx = BX + 0.18
    bw = BW - 0.36
    bh = BH_each
    R(bx, by, bw, bh, fc=BLU, ec=BLU_E, lw=1.5, rr=0.10, z=2)
    T(bx+bw/2, by+bh-0.28, label, fs=10.5, fw='bold')
    for i, line in enumerate(sub_lines):
        T(bx+bw/2, by+bh-0.62-i*0.34, line, fs=8.5)
    # SIEVE formula at bottom
    T(bx+bw/2, by+0.42, tau_line, fs=8, st='italic', c='#333')
    T(bx+bw/2, by+0.14, 'else  0', fs=8, st='italic', c='#333')

# SIFT1M  (top)
S1_Y = BOT + 2*(BH_each + BH_gap) + 0.60
bench_box('SIFT1M',
          ['1M vectors · 128-dim', 'L2 metric · ANN'],
          'Score = QPS  if  Recall@10 ≥ 0.90', S1_Y)

# GLDv2   (middle)
GL_Y = BOT + 1*(BH_each + BH_gap) + 0.60
bench_box('GLDv2',
          ['1.58M vectors · 512-dim', 'IP metric · Landmark retrieval'],
          'Score = QPS  if  mAP ≥ 0.20', GL_Y)

# HICO-DET  (bottom)
HC_Y = BOT + 0*(BH_each + BH_gap) + 0.20
bench_box('HICO-DET',
          ['47,776 images · 600 HOI classes', 'CLIP ViT-L/14 + DINOv2 rerank'],
          'Score = QPS  if  mAP ≥ 0.15', HC_Y)

# Arrows: each benchmark → loop panel
for by in [S1_Y, GL_Y, HC_Y]:
    my = by + BH_each/2
    arrow(BX+BW, my, LX, my, ec=BLU_E, lw=1.5)

# ════════════════════════════════════════════════════════════════════════════
# 2. OPTIMIZATION LOOP PANEL
# ════════════════════════════════════════════════════════════════════════════
R(LX, BOT, LW, H, fc=GRN, ec=GRN_E, lw=2.5, rr=0.15, z=1, alpha=0.45)
T(LX+LW/2, TOP-0.35, 'Hyperparameter Optimization Loop', fs=13, fw='bold')

# ── 2a. Methods sub-panel ────────────────────────────────────────────────────
MP_X = LX + 0.20;  MP_W = LW - 0.40
MP_Y = BOT + H*0.505;  MP_H = TOP - 0.55 - MP_Y
R(MP_X, MP_Y, MP_W, MP_H, fc=GRY, ec=GRY_E, lw=1.0, rr=0.10, z=2, alpha=0.95)
T(MP_X+MP_W/2, MP_Y+MP_H-0.26, 'Optimization Methods', fs=11, fw='bold')

# Method geometry
MBY = MP_Y + 0.18
MBH = MP_H - 0.50
GAP  = 0.22
# Column widths: LLM(wide) | GP-BO | Optuna | Grid+Random
# Available inner width
INNER = MP_W - 0.35*2
LLM_W  = INNER * 0.280
NORM_W = INNER * 0.225
GR_W   = INNER * 0.200
# x starts
X_LLM    = MP_X + 0.35
X_GPBO   = X_LLM  + LLM_W  + GAP
X_OPT    = X_GPBO + NORM_W  + GAP
X_GR     = X_OPT  + NORM_W  + GAP

def mbox(bx, by, bw, bh, title, lines, style='blue'):
    fc, ec = (BLU, BLU_E) if style=='blue' else (YEL, YEL_E)
    R(bx, by, bw, bh, fc=fc, ec=ec, lw=1.5, rr=0.11, z=3)
    T(bx+bw/2, by+bh-0.28, title, fs=10, fw='bold')
    for i, line in enumerate(lines):
        T(bx+bw/2, by+bh-0.62-i*0.34, line, fs=8.5)

# LLM Agent
mbox(X_LLM, MBY, LLM_W, MBH, 'LLM Agent',
     ['MiniMax-M2.1 (OpenRouter)', 'Phase-aware prompting', 'Dedup guard · 4 retries'])

# History Buffer H_t  (inside LLM box, bottom)
HBX = X_LLM + 0.12;  HBW = LLM_W - 0.24
HBH = MBH * 0.325;   HBY = MBY + 0.12
R(HBX, HBY, HBW, HBH, fc=HIST, ec=HIST_E, lw=2.0, rr=0.09, z=4)
T(HBX+HBW/2, HBY+HBH*0.66,
  r'$\mathcal{H}_t$  History Buffer', fs=9, fw='bold', c='#8B0000')
T(HBX+HBW/2, HBY+HBH*0.26,
  'past (config, score) pairs', fs=7.5, c='#555')

# GP-BO
mbox(X_GPBO, MBY, NORM_W, MBH, 'GP-BO',
     ['Gaussian Process', 'Bayesian Optim.', 'Acq.: Exp. Improvement'])

# Optuna TPE
mbox(X_OPT, MBY, NORM_W, MBH, 'Optuna TPE',
     ['Bayesian', '(Tree Parzen Est.)', 'Dir.: maximize'])

# Grid Search  (top half of GR column)
GRD_H = MBH * 0.54;  RND_H = MBH - GRD_H - 0.22
mbox(X_GR, MBY+RND_H+0.22, GR_W, GRD_H, '⊞  Grid Search',
     ['50 principled configs', 'Deterministic'], style='yellow')

# Random Search  (bottom half)
mbox(X_GR, MBY, GR_W, RND_H, '∼  Random Search',
     ['Uniform sampling'], style='yellow')

# ── 2b. Config Proposal ──────────────────────────────────────────────────────
CFG_Y = BOT + H*0.310;  CFG_H = H*0.175
CFG_X = LX + 0.80;      CFG_W = LW - 1.60
R(CFG_X, CFG_Y, CFG_W, CFG_H, fc=YEL, ec=YEL_E, lw=1.5, rr=0.12, z=3)
T(CFG_X+CFG_W/2, CFG_Y+CFG_H-0.28, '≡  Config Proposal', fs=10.5, fw='bold')
T(CFG_X+CFG_W/2, CFG_Y+CFG_H-0.64,
  'engine ∈ { FaissFlat  |  FaissHNSWFlat }', fs=9.5)
T(CFG_X+CFG_W/2, CFG_Y+CFG_H-1.00,
  'M ∈ [8, 64]  ·  efConstruction ∈ [100, 500]  ·  efSearch ∈ [16, 512]', fs=9)
T(CFG_X+CFG_W/2, CFG_Y+0.22,
  '(FaissFlat: no extra params  ·  FaissIVFFlat excluded: probes all cells at 1M scale)',
  fs=7.5, c='#666', st='italic')

# Arrows: every method box → top of Config Proposal
for mx in [X_LLM+LLM_W/2, X_GPBO+NORM_W/2, X_OPT+NORM_W/2, X_GR+GR_W/2]:
    arrow(mx, MBY, mx, CFG_Y+CFG_H, ec=BLU_E, lw=1.2, z=4)

# ── 2c. Phase bar ─────────────────────────────────────────────────────────────
PH_Y = BOT + H*0.168;  PH_H = H*0.118
PH_X = LX + 0.20;      PH_W = LW - 0.40

phases = [
    ('↗  EXPLORATION',  'iter 1–20',  BLU, BLU_E, 20/50),
    ('⊙  EXPLOITATION', 'iter 21–38', GRN, GRN_E, 18/50),
    ('∇  FINE-TUNING',  'iter 39–50', YEL, YEL_E, 12/50),
]
cx = PH_X
for name, iters, fc, ec, frac in phases:
    pw = PH_W * frac
    R(cx, PH_Y, pw, PH_H, fc=fc, ec=ec, lw=1.5, rr=0.0, z=3)
    T(cx+pw/2, PH_Y+PH_H*0.64, name, fs=9, fw='bold')
    T(cx+pw/2, PH_Y+PH_H*0.22, iters, fs=8.5, c='#333')
    cx += pw
T(PH_X+PH_W/2, PH_Y-0.22,
  '(phase-aware adaptive prompting — LLM Agent only)',
  fs=7.5, c='#555', st='italic')

# Dashed: Config → phase bar
arrow(CFG_X+CFG_W/2, CFG_Y, CFG_X+CFG_W/2, PH_Y+PH_H,
      ec='#aaa', lw=1.0, dashed=True, lbl='iteration i', lfs=8, lbl_dy=0.20, z=4)

# ── 2d. Best Configuration ────────────────────────────────────────────────────
BST_Y = BOT + 0.18;  BST_H = H*0.130
BST_X = LX + 0.80;  BST_W = LW - 1.60
R(BST_X, BST_Y, BST_W, BST_H, fc=GRN, ec=GRN_E, lw=1.5, rr=0.12, z=3)
T(BST_X+BST_W/2, BST_Y+BST_H-0.30, '★  Best Configuration', fs=10.5, fw='bold')
T(BST_X+BST_W/2, BST_Y+BST_H*0.36,
  'argmax  Score  over all 50 iterations', fs=9)

# Dashed: phase → best
arrow(PH_X+PH_W/2, PH_Y, PH_X+PH_W/2, BST_Y+BST_H,
      ec='#aaa', lw=1.0, dashed=True, z=4)

# Main flow: Config → VDMS  (exits right side of config box)
CFG_ARROW_Y = CFG_Y + CFG_H * 0.55
arrow(CFG_X+CFG_W, CFG_ARROW_Y, VX, CFG_ARROW_Y,
      ec=GRN_E, lw=2.5, lbl='index config', lfs=8.5, lbl_dy=0.20, z=5)

# ════════════════════════════════════════════════════════════════════════════
# 3. VDMS PANEL
# ════════════════════════════════════════════════════════════════════════════
# VDMS container (top)
VDMS_Y = BOT + H*0.445;  VDMS_H = TOP - 0.20 - VDMS_Y
VDMS_X = VX + 0.10;      VDMS_W = VW - 0.20
R(VDMS_X, VDMS_Y, VDMS_W, VDMS_H, fc=VBKG, ec=VBKG_E, lw=2.0, rr=0.12, z=2)
T(VDMS_X+VDMS_W/2, VDMS_Y+VDMS_H-0.30,
  'VDMS  (Apptainer)', fs=11, fw='bold')

# Vector Store (dark green block)
VS_X = VDMS_X + 0.25;  VS_W = VDMS_W - 0.50
VS_Y = VDMS_Y + VDMS_H*0.34;  VS_H = VDMS_H * 0.56
R(VS_X, VS_Y, VS_W, VS_H, fc=GRN_E, ec='#1a5c1a', lw=1.5, rr=0.10, z=3)
T(VS_X+VS_W/2, VS_Y+VS_H*0.60, 'Vector Store', fs=10.5, fw='bold', c='white')
T(VS_X+VS_W/2, VS_Y+VS_H*0.26, '(batch ≤ 50 per AddDescriptor)', fs=8, c='#ddd')

# Engine rows
ENG_X = VDMS_X + 0.18;  ENG_W = VDMS_W - 0.36
ROW_H = 0.50;  ROW_GAP = 0.08
E_BOT = VDMS_Y + 0.15

# FaissIVFFlat  (bottom — de-emphasised)
R(ENG_X, E_BOT, ENG_W, ROW_H, fc=GRY, ec=GRY_E, lw=1.0, rr=0.07, z=4)
T(ENG_X+ENG_W/2, E_BOT+ROW_H/2,
  'FaissIVFFlat  (nlist / nprobe)', fs=8.5, c='#999')

# FaissHNSWFlat  (middle — highlighted, winner)
R(ENG_X, E_BOT+ROW_H+ROW_GAP, ENG_W, ROW_H,
  fc=HNSW, ec=HNSW_E, lw=2.2, rr=0.07, z=4)
T(ENG_X+ENG_W/2, E_BOT+ROW_H+ROW_GAP+ROW_H/2,
  'FaissHNSWFlat  ★  M / efC / efS', fs=8.5, fw='bold', c='#1a5c1a')

# FaissFlat  (top — exact baseline)
R(ENG_X, E_BOT+2*(ROW_H+ROW_GAP), ENG_W, ROW_H,
  fc=GRY, ec=GRY_E, lw=1.0, rr=0.07, z=4)
T(ENG_X+ENG_W/2, E_BOT+2*(ROW_H+ROW_GAP)+ROW_H/2,
  'FaissFlat  (exact · recall ceiling)', fs=8.5)

# ── Benchmark Evaluation box ──────────────────────────────────────────────────
EVAL_Y = BOT + H*0.26;  EVAL_H = H*0.160
EVAL_X = VX + 0.10;     EVAL_W = VW - 0.20
R(EVAL_X, EVAL_Y, EVAL_W, EVAL_H, fc=BLU, ec=BLU_E, lw=1.5, rr=0.12, z=3)
T(EVAL_X+EVAL_W/2, EVAL_Y+EVAL_H-0.28, '▶  Benchmark Evaluation', fs=10, fw='bold')
T(EVAL_X+EVAL_W/2, EVAL_Y+EVAL_H-0.62, '① Build ANN index  (batch = 50)', fs=9)
T(EVAL_X+EVAL_W/2, EVAL_Y+EVAL_H-0.93, '② Run  Nₚ  queries', fs=9)
T(EVAL_X+EVAL_W/2, EVAL_Y+EVAL_H-1.24, '③ Measure QPS + Recall@10 / mAP', fs=9)

# Arrow VDMS → Benchmark Eval
arrow(VDMS_X+VDMS_W/2, VDMS_Y,
      EVAL_X+EVAL_W/2,  EVAL_Y+EVAL_H,
      ec=GRN_E, lw=2.0, lbl='index + queries', lfs=8, lbl_dx=0.7, lbl_dy=0, z=5)

# ── Score diamond ─────────────────────────────────────────────────────────────
SCR_CX = EVAL_X + EVAL_W/2
SCR_CY = BOT + H*0.097
SCR_W  = EVAL_W * 0.90
SCR_H  = H * 0.165

arrow(EVAL_X+EVAL_W/2, EVAL_Y, SCR_CX, SCR_CY+SCR_H/2,
      ec=ORN_E, lw=2.0, z=5)

diamond(SCR_CX, SCR_CY, SCR_W, SCR_H, fc=ORN, ec=ORN_E, lw=2.0, z=3)
T(SCR_CX, SCR_CY+0.24, '◆  SIEVE Score', fs=10, fw='bold')
T(SCR_CX, SCR_CY-0.10, 'QPS  if metric ≥ τ', fs=9)
T(SCR_CX, SCR_CY-0.38, 'else  0', fs=9)

# ── Score → Best (dashed green, right side of diamond → right of best box) ────
# Right point of diamond
SCR_RX = SCR_CX + SCR_W/2
SCR_RY = SCR_CY
routed_arrow(
    [(SCR_RX, SCR_RY),
     (VX + VW - 0.05, SCR_RY),
     (VX + VW - 0.05, BST_Y + BST_H/2),
     (BST_X + BST_W,  BST_Y + BST_H/2)],
    ec=GRN_E, lw=1.5, dashed=True,
    lbl='best result', lfs=8, lbl_idx=2, lbl_dx=0.6, lbl_dy=0, z=5)

# ── Feedback: Score → LLM History Buffer (dashed red, routes under figure) ────
SCR_LX = SCR_CX - SCR_W/2
HT_MX  = HBX + HBW/2
HT_Y   = HBY + HBH/2
ROUTE_Y = BOT - 0.05   # just below panel bottom
ROUTE_X = BX - 0.08    # just left of benchmarks

routed_arrow(
    [(SCR_LX,  SCR_CY),
     (ROUTE_X, SCR_CY),
     (ROUTE_X, HT_Y),
     (HBX,     HT_Y)],
    ec=REDD, lw=2.0, dashed=True,
    lbl='feedback (LLM)', lfs=8.5, lbl_idx=2, lbl_dx=0.7, lbl_dy=0, z=5)

# ── Feedback: Score → GP-BO / Optuna (dashed red, routes over top) ────────────
FB_TOP = TOP + 0.08
GP_MX  = X_GPBO + NORM_W/2
OPT_MX = X_OPT  + NORM_W/2
FB_MY  = MBY + MBH   # top of method boxes

# Route from right edge of Score diamond up and then left to GP-BO + Optuna
routed_arrow(
    [(SCR_RX,   SCR_CY),
     (VX+VW+0.08, SCR_CY),
     (VX+VW+0.08, FB_TOP),
     (OPT_MX,    FB_TOP),
     (OPT_MX,    FB_MY)],
    ec=REDD, lw=1.5, dashed=True, z=5)
# second branch for GP-BO
routed_arrow(
    [(VX+VW+0.08, FB_TOP),
     (GP_MX,      FB_TOP),
     (GP_MX,      FB_MY)],
    ec=REDD, lw=1.5, dashed=True, z=5)
# label on the top horizontal
T((VX+VW+0.08 + OPT_MX)/2, FB_TOP+0.18,
  'feedback (seq. opt.)', fs=8, c=REDD)

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
LG_X = BX + 0.05;  LG_Y = BOT + 0.10
LG_W = BW - 0.10;  LG_H = 1.55
R(LG_X, LG_Y, LG_W, LG_H, fc='white', ec='#cccccc', lw=1, rr=0.08, z=8)
T(LG_X+LG_W/2, LG_Y+LG_H-0.22, 'Legend', fs=9, fw='bold')

legend_items = [
    (BLU,  BLU_E,  'Adaptive optimizer / benchmark'),
    (YEL,  YEL_E,  'Baseline optimizer / proposal'),
    (GRN,  GRN_E,  'VDMS / system component'),
    (HIST, HIST_E, 'History Buffer (LLM only)'),
]
for i, (fc, ec, lbl) in enumerate(legend_items):
    iy = LG_Y + LG_H - 0.52 - i*0.28
    R(LG_X+0.10, iy-0.09, 0.35, 0.22, fc=fc, ec=ec, lw=1, rr=0.04, z=9)
    T(LG_X+0.55, iy, lbl, fs=7.5, ha='left', z=9)

# Dashed feedback note
ax.plot([LG_X+0.10, LG_X+0.45], [LG_Y+0.15, LG_Y+0.15],
        color=REDD, lw=1.5, linestyle=(0,(5,3)), zorder=9)
T(LG_X+0.55, LG_Y+0.15, 'Feedback arrow', fs=7.5, ha='left', z=9)

# ════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, bbox_inches='tight', dpi=200, format='pdf')
print(f'Saved: {OUT}')
