#!/usr/bin/env python3
"""
Generate Figure 2 — faithful recreation of the April-17 drawio style
with only the 10 required content fixes applied.
Saves: figures/workflow.pdf
"""
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'workflow.pdf')

# ── Canvas ───────────────────────────────────────────────────────────────────
# ylim extends above and below panels for wraparound feedback arrows
FW, FH   = 18.0, 9.5
YMIN, YMAX = -1.0, 9.5     # data range
fig, ax  = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(YMIN, YMAX)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Exact drawio colours ─────────────────────────────────────────────────────
BLU,  BLU_E  = '#dae8fc', '#6c8ebf'
YEL,  YEL_E  = '#fff2cc', '#d6b656'
GRN,  GRN_E  = '#d5e8d4', '#82b366'
ORN,  ORN_E  = '#ffe6cc', '#d79b00'
GRY,  GRY_E  = '#f5f5f5', '#666666'
VS_F, VS_E   = '#6d8764', '#4d6144'
HNSW, HNSW_E = '#cbf0cb', '#1a7a1a'
HIST, HIST_E = '#fce4d6', '#c0392b'
FBARR        = '#b85450'
DASH_PAT     = (0, (7, 3))

# ── Panels: y=[0.30, 7.65]  ───────────────────────────────────────────────────
BOT, TOP = 0.30, 7.65
H = TOP - BOT      # 7.35

BX, BW = 0.12, 2.78    # Benchmarks
LX, LW = 3.02, 9.86    # Loop
VX, VW = 13.00, 4.78   # VDMS column

# ── Helpers ──────────────────────────────────────────────────────────────────
def R(x, y, w, h, fc=GRY, ec=GRY_E, lw=1.5, rr=0.10, z=3, alpha=1.0,
      ls='solid'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={rr}',
                       fc=fc, ec=ec, lw=lw, linestyle=ls,
                       zorder=z, alpha=alpha, clip_on=False)
    ax.add_patch(p)

def T(x, y, s, fs=9, fw='normal', ha='center', va='center',
      c='black', z=8, st='normal', rotation=0):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, fontstyle=st,
            ha=ha, va=va, color=c, zorder=z, clip_on=False, rotation=rotation)

def polyline(pts, ec, lw=1.5, dashed=False, z=5):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=ec, lw=lw, zorder=z, clip_on=False,
            linestyle=DASH_PAT if dashed else 'solid',
            solid_capstyle='butt')

def arrowhead(tip, prev, ec, lw=1.5, z=6):
    ax.annotate('', xy=tip, xytext=prev,
                arrowprops=dict(arrowstyle='->', color=ec, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=z, annotation_clip=False)

def routed(pts, ec, lw=1.5, dashed=False, z=5):
    polyline(pts, ec, lw, dashed, z)
    arrowhead(pts[-1], pts[-2], ec, lw, z)

def diamond(cx, cy, w, h, fc=ORN, ec=ORN_E, lw=2.0, z=3):
    verts = [[cx, cy+h/2], [cx+w/2, cy], [cx, cy-h/2], [cx-w/2, cy]]
    ax.add_patch(Polygon(verts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  BENCHMARKS PANEL
# ═══════════════════════════════════════════════════════════════════════════════
R(BX, BOT, BW, H, fc='none', ec='#888888', lw=1.5, rr=0.15, z=1,
  ls=DASH_PAT)
T(BX+BW/2, TOP-0.30, 'Benchmarks', fs=12, fw='bold')

# Three boxes stacked evenly, clear of the title
TITLE_GAP = 0.70        # space reserved at top for "Benchmarks" label
BOT_GAP   = 0.20        # small bottom margin inside panel
B_GAP     = 0.18        # gap between boxes
AVAIL_B   = H - TITLE_GAP - BOT_GAP - 2*B_GAP
BH3       = AVAIL_B / 3                    # ≈ 2.06 in
BI_X      = BX + 0.15
BI_W      = BW - 0.30

HIC_Y  = BOT + BOT_GAP
GLD_Y  = HIC_Y  + BH3 + B_GAP
SIFT_Y = GLD_Y  + BH3 + B_GAP

def bench(label, line1, line2, score_line, by):
    R(BI_X, by, BI_W, BH3, fc=BLU, ec=BLU_E, lw=1.5, rr=0.10, z=2)
    cx = BI_X + BI_W/2
    T(cx, by+BH3-0.28, label,      fs=10.5, fw='bold')
    T(cx, by+BH3-0.58, line1,      fs=8)
    if line2:
        T(cx, by+BH3-0.85, line2,  fs=8)
    T(cx, by+0.46,     score_line, fs=7.5, st='italic', c='#333')
    T(cx, by+0.18,     'else  0',  fs=7.5, st='italic', c='#333')

bench('SIFT1M',
      '1,000,000 vectors · 128-dim',
      'L2 metric · ANN benchmark',
      'Score = QPS  if  Recall@10 ≥ 0.90', SIFT_Y)

bench('GLDv2',
      '1,580,470 vectors · 512-dim',
      'IP metric · Landmark retrieval',
      'Score = QPS  if  mAP ≥ 0.20', GLD_Y)

bench('HICO-DET',
      '47,776 images · 600 HOI classes',
      'CLIP ViT-L/14 + DINOv2 rerank',
      'Score = QPS  if  mAP ≥ 0.15', HIC_Y)

# Arrows from benchmark centres into loop panel left edge
for by in [SIFT_Y, GLD_Y, HIC_Y]:
    cy = by + BH3/2
    routed([(BX+BW, cy), (LX, cy)], BLU_E, lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  OPTIMIZATION LOOP PANEL
# ═══════════════════════════════════════════════════════════════════════════════
R(LX, BOT, LW, H, fc=GRN, ec=GRN_E, lw=2.0, rr=0.15, z=1, alpha=0.35)
T(LX+LW/2, TOP-0.30, 'Hyperparameter Optimization Loop',
  fs=12.5, fw='bold', c='#3d6600')

# ── 2a. Methods sub-panel ─────────────────────────────────────────────────────
MPX = LX+0.18; MPW = LW-0.36
MPY = BOT + H*0.510; MPH = TOP - 0.52 - MPY
R(MPX, MPY, MPW, MPH, fc='white', ec=GRN_E, lw=1.0, rr=0.10, z=2, alpha=0.9)
T(MPX+MPW/2, MPY+MPH-0.24, 'Optimization Methods', fs=11, fw='bold')

MBY = MPY + 0.12
MBH = MPH - 0.46

# Column widths proportional to content
GAP    = 0.18
INNER  = MPW - 0.28
# LLM takes ~30%, GP-BO ~22%, Optuna ~22%, Grid/Rand ~26%
LLM_W  = INNER * 0.285; NOm_W = INNER * 0.220; GRD_W = INNER * 0.255
# renormalize so they exactly fill INNER - 3*GAP
used   = LLM_W + NOm_W + NOm_W + GRD_W + 3*GAP
s      = (INNER - 3*GAP) / (LLM_W + 2*NOm_W + GRD_W)
LLM_W *= s; NOm_W *= s; GRD_W *= s

X0     = MPX + 0.14
X_LLM  = X0
X_GPBO = X_LLM  + LLM_W + GAP
X_OPT  = X_GPBO + NOm_W + GAP
X_GRD  = X_OPT  + NOm_W + GAP

def mbox(bx, bw, title, lines, style='blue'):
    fc, ec = (BLU, BLU_E) if style=='blue' else (YEL, YEL_E)
    R(bx, MBY, bw, MBH, fc=fc, ec=ec, lw=1.5, rr=0.10, z=3)
    cx = bx + bw/2
    T(cx, MBY+MBH-0.26, title, fs=9.5, fw='bold')
    for i, ln in enumerate(lines):
        T(cx, MBY+MBH-0.56-i*0.32, ln, fs=8)

mbox(X_LLM,  LLM_W, 'LLM Agent',
     ['MiniMax-M2.1 (OpenRouter)', 'Phase-aware prompting',
      'Dedup guard · 4 retries'])

# History Buffer H_t  — inside LLM Agent, at the bottom
HBM = 0.12
HBX = X_LLM + HBM;  HBW = LLM_W - 2*HBM
HBH = MBH * 0.30;   HBY = MBY + HBM
R(HBX, HBY, HBW, HBH, fc=HIST, ec=HIST_E, lw=1.8, rr=0.08, z=4)
HT_CX = HBX + HBW/2;  HT_CY = HBY + HBH/2
T(HT_CX, HBY+HBH*0.68, r'$\mathcal{H}_t$  History Buffer',
  fs=8.5, fw='bold', c='#8B0000')
T(HT_CX, HBY+HBH*0.28, 'past (config, score) pairs', fs=7.5, c='#555')

mbox(X_GPBO, NOm_W, 'GP-BO',
     ['Gaussian Process', 'Bayesian Optim.', 'Exp. Improvement'])

mbox(X_OPT,  NOm_W, '⊕ Optuna TPE',
     ['Bayesian (Tree Parzen)', 'Direction: maximize'])

# Grid Search (top half of right column)
GRD_H = MBH * 0.52;  RND_H = MBH - GRD_H - 0.16
R(X_GRD, MBY+RND_H+0.16, GRD_W, GRD_H, fc=YEL, ec=YEL_E, lw=1.5, rr=0.10, z=3)
GCX = X_GRD+GRD_W/2
T(GCX, MBY+RND_H+0.16+GRD_H-0.26, '⊞ Grid Search', fs=9.5, fw='bold')
T(GCX, MBY+RND_H+0.16+GRD_H-0.55, '50 principled configs', fs=8)
T(GCX, MBY+RND_H+0.16+GRD_H-0.82, 'Deterministic', fs=8)

# Random Search (bottom half)
R(X_GRD, MBY, GRD_W, RND_H, fc=YEL, ec=YEL_E, lw=1.5, rr=0.10, z=3)
T(X_GRD+GRD_W/2, MBY+RND_H*0.62, '∼ Random Search', fs=9.5, fw='bold')
T(X_GRD+GRD_W/2, MBY+RND_H*0.25, 'Uniform sampling', fs=8)

# Arrows: each method bottom → Config Proposal top
for cx in [X_LLM+LLM_W/2, X_GPBO+NOm_W/2, X_OPT+NOm_W/2, X_GRD+GRD_W/2]:
    routed([(cx, MBY), (cx, BOT+H*0.338)], BLU_E, lw=1.2)

# ── 2b. Config Proposal ────────────────────────────────────────────────────────
CFG_X = LX+0.75;  CFG_W = LW-1.50
CFG_Y = BOT+H*0.338; CFG_H = H*0.170
R(CFG_X, CFG_Y, CFG_W, CFG_H, fc=YEL, ec=YEL_E, lw=1.5, rr=0.12, z=3)
CC = CFG_X+CFG_W/2
T(CC, CFG_Y+CFG_H-0.26, '≡  Config Proposal', fs=10.5, fw='bold')
T(CC, CFG_Y+CFG_H-0.58, '{ engine: FaissFlat  |  FaissHNSWFlat }', fs=9.5)
T(CC, CFG_Y+CFG_H-0.88,
  'M ∈ [8,64]  ·  efConstruction ∈ [100,500]  ·  efSearch ∈ [16,512]', fs=9)
T(CC, CFG_Y+0.22,
  '(FaissFlat: no extra params   ·   FaissIVFFlat: excluded at 1M scale)',
  fs=7.5, c='#666', st='italic')

# ── 2c. Phase bar ──────────────────────────────────────────────────────────────
PH_X = LX+0.18;  PH_W = LW-0.36
PH_Y = BOT+H*0.172; PH_H = H*0.115
segs = [
    ('↗ EXPLORATION',  'iter 1–20',  BLU, BLU_E, 20/50),
    ('⊙ EXPLOITATION', 'iter 21–38', GRN, GRN_E, 18/50),
    ('∇ FINE-TUNING',  'iter 39–50', YEL, YEL_E, 12/50),
]
cx = PH_X
for name, itr, fc, ec, frac in segs:
    pw = PH_W * frac
    R(cx, PH_Y, pw, PH_H, fc=fc, ec=ec, lw=1.5, rr=0.0, z=3)
    T(cx+pw/2, PH_Y+PH_H*0.65, name, fs=9, fw='bold')
    T(cx+pw/2, PH_Y+PH_H*0.22, itr,  fs=8.5, c='#333')
    cx += pw
T(PH_X+PH_W/2, PH_Y-0.20,
  '(phase-aware adaptive prompting — LLM only)', fs=7.5, c='#555', st='italic')

# Dashed: Config → phase bar
routed([(CC, CFG_Y), (CC, PH_Y+PH_H)], '#aaaaaa', lw=1.0, dashed=True)
T(CC+0.18, (CFG_Y+PH_Y+PH_H)/2, 'iteration i', fs=7.5, c='#888', ha='left')

# ── 2d. Best Configuration ─────────────────────────────────────────────────────
BST_X = LX+0.75;  BST_W = LW-1.50
BST_Y = BOT+0.12; BST_H = H*0.130
R(BST_X, BST_Y, BST_W, BST_H, fc=GRN, ec=GRN_E, lw=1.5, rr=0.12, z=3)
BC = BST_X+BST_W/2
T(BC, BST_Y+BST_H-0.28, '★  Best Configuration', fs=10.5, fw='bold')
T(BC, BST_Y+BST_H*0.35, 'argmax  Score  over all 50 iterations', fs=9)

# Dashed: phase → best
routed([(PH_X+PH_W/2, PH_Y), (PH_X+PH_W/2, BST_Y+BST_H)],
       '#aaaaaa', lw=1.0, dashed=True)

# ── Config → VDMS  (exits right, routes up inside loop panel, enters VDMS left)
CFG_EXIT_Y  = CFG_Y + CFG_H/2
# Pre-compute VDMS geometry (same as Section 3 below)
_VDMS_Y = BOT + H*0.455
_VDMS_H = TOP - 0.22 - _VDMS_Y
VDMS_ENTRY_Y = _VDMS_Y + _VDMS_H * 0.62   # enters at Vector Store level
# Vertical routing column: just inside the loop panel right edge,
# clear of the Grid/Random column which ends at ~LX+LW-0.40
ROUTE_X_V = LX + LW - 0.28
routed([
    (CFG_X+CFG_W, CFG_EXIT_Y),
    (ROUTE_X_V,   CFG_EXIT_Y),
    (ROUTE_X_V,   VDMS_ENTRY_Y),
    (VX+0.15,     VDMS_ENTRY_Y),
], GRN_E, lw=2.0)
# Rotated label on the vertical segment (standard for tight routing columns)
T(ROUTE_X_V + 0.18, (CFG_EXIT_Y + VDMS_ENTRY_Y)/2,
  'index config', fs=8, c='#4a7a4a', ha='center', va='center', rotation=90)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  VDMS PANEL
# ═══════════════════════════════════════════════════════════════════════════════
VDMS_Y = BOT + H*0.455;  VDMS_H = TOP - 0.22 - VDMS_Y
VDMS_X = VX + 0.15;      VDMS_W = VW - 0.30

R(VDMS_X, VDMS_Y, VDMS_W, VDMS_H, fc=GRN, ec=GRN_E, lw=2.0, rr=0.12, z=2)
T(VDMS_X+VDMS_W/2, VDMS_Y+VDMS_H-0.28,
  'VDMS  (Apptainer container)', fs=11, fw='bold')

# Vector Store
VS_X = VDMS_X+0.22; VS_W = VDMS_W-0.44
VS_Y = VDMS_Y+VDMS_H*0.35; VS_H = VDMS_H*0.53
R(VS_X, VS_Y, VS_W, VS_H, fc=VS_F, ec=VS_E, lw=1.5, rr=0.10, z=3)
T(VS_X+VS_W/2, VS_Y+VS_H*0.58, 'Vector Store', fs=11, fw='bold', c='white')
T(VS_X+VS_W/2, VS_Y+VS_H*0.24, '(batch ≤ 50 per AddDescriptor)', fs=8, c='#ddd')

# Engine rows
EX = VDMS_X+0.22; EW = VDMS_W-0.44
RH = 0.40; RG = 0.07
E_BOT = VDMS_Y + 0.14

R(EX, E_BOT+2*(RH+RG), EW, RH, fc=GRY,  ec=GRY_E,  lw=1.0, rr=0.06, z=4)
T(EX+EW/2, E_BOT+2*(RH+RG)+RH/2,
  'FaissFlat  (exact · recall ceiling)', fs=8.5)

R(EX, E_BOT+(RH+RG), EW, RH, fc=HNSW, ec=HNSW_E, lw=2.0, rr=0.06, z=4)
T(EX+EW/2, E_BOT+(RH+RG)+RH/2,
  'FaissHNSWFlat  ★  M · efC · efS', fs=8.5, fw='bold', c='#1a5c1a')

R(EX, E_BOT, EW, RH, fc=GRY, ec='#bbbbbb', lw=1.0, rr=0.06, z=4)
T(EX+EW/2, E_BOT+RH/2,
  'FaissIVFFlat  (nlist / nprobe)', fs=8.5, c='#999')

# ── Benchmark Evaluation ───────────────────────────────────────────────────────
EVAL_X = VX+0.15; EVAL_W = VW-0.30
EVAL_Y = BOT+H*0.265; EVAL_H = H*0.160
R(EVAL_X, EVAL_Y, EVAL_W, EVAL_H, fc=BLU, ec=BLU_E, lw=1.5, rr=0.12, z=3)
EC_X = EVAL_X+EVAL_W/2
T(EC_X, EVAL_Y+EVAL_H-0.28, '▶  Benchmark Evaluation', fs=10, fw='bold')
T(EC_X, EVAL_Y+EVAL_H-0.60, '① Build ANN index  (batch=50)', fs=9)
T(EC_X, EVAL_Y+EVAL_H-0.90, r'② Run  $N_q$  queries', fs=9)
T(EC_X, EVAL_Y+EVAL_H-1.20, '③ Measure QPS + Recall@10 / mAP', fs=9)

# VDMS → Eval
routed([(VDMS_X+VDMS_W/2, VDMS_Y),
        (VDMS_X+VDMS_W/2, EVAL_Y+EVAL_H)], GRN_E, lw=1.8)
T(VDMS_X+VDMS_W/2+0.14, (VDMS_Y+EVAL_Y+EVAL_H)/2,
  'index + queries', fs=8, c='#4a7a4a', ha='left')

# ── Score diamond ──────────────────────────────────────────────────────────────
SCR_CX = EVAL_X+EVAL_W/2
SCR_CY = BOT + H*0.095
SCR_W  = EVAL_W*0.86
SCR_H  = H*0.155

routed([(EC_X, EVAL_Y), (EC_X, SCR_CY+SCR_H/2)], ORN_E, lw=1.8)
diamond(SCR_CX, SCR_CY, SCR_W, SCR_H, fc=ORN, ec=ORN_E, lw=2.0, z=3)
T(SCR_CX, SCR_CY+0.26,  '◆  Score  (SIEVE)', fs=10, fw='bold')
T(SCR_CX, SCR_CY-0.08,  'QPS  if  metric ≥ τ', fs=9)
T(SCR_CX, SCR_CY-0.36,  'else  0', fs=9)

# Score → Best Config  (left of diamond → right of Best Config, same y)
SCR_LEFT  = SCR_CX - SCR_W/2
SCR_BOT   = SCR_CY - SCR_H/2
BST_RIGHT = BST_X + BST_W
BST_MID_Y = BST_Y + BST_H/2
routed([
    (SCR_LEFT, SCR_CY),
    (BST_RIGHT+0.10, SCR_CY),
    (BST_RIGHT+0.10, BST_MID_Y),
    (BST_RIGHT, BST_MID_Y),
], GRN_E, lw=1.5, dashed=True)
T(BST_RIGHT+0.25, (SCR_CY+BST_MID_Y)/2,
  'best result', fs=8, c='#4a7a4a', ha='left')

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FEEDBACK ARROWS  (route around the outside of all panels)
# ═══════════════════════════════════════════════════════════════════════════════

# ── LLM feedback: Score bottom → below panels → left → up → H_t ──────────────
FB_BOT   = -0.60         # routing y below all panels
FB_LEFT  = -0.12         # routing x left of all panels (clip_on=False, OK)
HT_LEFT  = HBX           # H_t box left edge
HT_MID_Y = HBY + HBH/2  # H_t vertical centre

polyline([(SCR_CX, SCR_BOT), (SCR_CX, FB_BOT)], FBARR, lw=1.8, dashed=True)
polyline([(SCR_CX, FB_BOT),  (FB_LEFT, FB_BOT)], FBARR, lw=1.8, dashed=True)
polyline([(FB_LEFT, FB_BOT), (FB_LEFT, HT_MID_Y)], FBARR, lw=1.8, dashed=True)
polyline([(FB_LEFT, HT_MID_Y), (HT_LEFT, HT_MID_Y)], FBARR, lw=1.8, dashed=True)
arrowhead((HT_LEFT, HT_MID_Y), (FB_LEFT, HT_MID_Y), FBARR, lw=1.8)
# Label centred on the bottom horizontal segment — always in visible area
T((SCR_CX + FB_LEFT) / 2, FB_BOT - 0.22,
  'feedback (LLM)', fs=9, c=FBARR, ha='center')

# ── Optuna / GP-BO feedback: Score right → outside right → over top ───────────
SCR_RIGHT = SCR_CX + SCR_W/2
FB_TOP    = TOP + 0.50       # routing y above all panels
FB_RIGHT  = VX + VW + 0.18  # routing x right of VDMS panel

GPBO_MX   = X_GPBO + NOm_W/2
OPT_MX    = X_OPT  + NOm_W/2
MB_TOP    = MBY + MBH

# shared path right → up → over top → horizontal
polyline([(SCR_RIGHT, SCR_CY), (FB_RIGHT,  SCR_CY)], FBARR, lw=1.5, dashed=True)
polyline([(FB_RIGHT,  SCR_CY), (FB_RIGHT,  FB_TOP)], FBARR, lw=1.5, dashed=True)
polyline([(FB_RIGHT,  FB_TOP), (GPBO_MX,   FB_TOP)], FBARR, lw=1.5, dashed=True)

# drop into GP-BO
polyline([(GPBO_MX, FB_TOP), (GPBO_MX, MB_TOP)], FBARR, lw=1.5, dashed=True)
arrowhead((GPBO_MX, MB_TOP), (GPBO_MX, FB_TOP), FBARR, lw=1.5)

# branch and drop into Optuna
polyline([(GPBO_MX, FB_TOP), (OPT_MX, FB_TOP)], FBARR, lw=1.5, dashed=True)
polyline([(OPT_MX,  FB_TOP), (OPT_MX, MB_TOP)], FBARR, lw=1.5, dashed=True)
arrowhead((OPT_MX, MB_TOP), (OPT_MX, FB_TOP), FBARR, lw=1.5)

T((GPBO_MX+OPT_MX)/2, FB_TOP+0.18,
  'feedback (Optuna, GP-BO)', fs=9, c=FBARR)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LEGEND  (horizontal strip below panels)
# ═══════════════════════════════════════════════════════════════════════════════
LG_Y  = -0.88;  LG_H = 0.52
LG_X  = 0.15;   LG_W = FW - 0.30
R(LG_X, LG_Y, LG_W, LG_H, fc='white', ec='#cccccc', lw=1.0, rr=0.06, z=8)

items = [
    (BLU,  BLU_E,  'Adaptive optimizer / benchmark'),
    (YEL,  YEL_E,  'Baseline optimizer'),
    (GRN,  GRN_E,  'VDMS / system component'),
    (HIST, HIST_E, 'History Buffer (LLM only)'),
]
SW = 0.32; SH = 0.20   # swatch size
total_items = len(items) + 1   # +1 for dashed line entry
item_w = LG_W / total_items
for i, (fc, ec, lbl) in enumerate(items):
    ix = LG_X + i*item_w + item_w*0.05
    iy = LG_Y + LG_H/2
    R(ix, iy-SH/2, SW, SH, fc=fc, ec=ec, lw=1, rr=0.03, z=9)
    T(ix+SW+0.08, iy, lbl, fs=8, ha='left', z=9)

# Dashed red swatch
di = len(items)
dx = LG_X + di*item_w + item_w*0.05
dy = LG_Y + LG_H/2
ax.plot([dx, dx+SW], [dy, dy], color=FBARR, lw=1.5,
        linestyle=DASH_PAT, zorder=9, clip_on=False)
T(dx+SW+0.08, dy, 'Score feedback (LLM + seq. opt.)', fs=8, ha='left', z=9)

# ═══════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, bbox_inches='tight', dpi=200, format='pdf')
print(f'Saved: {OUT}')
