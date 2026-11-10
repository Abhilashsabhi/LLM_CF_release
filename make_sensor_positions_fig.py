#!/usr/bin/env python3
"""
make_sensor_positions_fig.py
Top-down car diagram with X/Y axes at each sensor. Car faces RIGHT.
FLU: X=forward(→right screen), Y=left(→up screen).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

OUT = "paper/sensor_positions_fig.pdf"

def sensor_axes(yaw_cw_deg):
    t = np.radians(yaw_cw_deg)
    return (np.array([ np.cos(t), -np.sin(t)]),
            np.array([ np.sin(t),  np.cos(t)]))

SENSORS = [
    ("CAM_FRONT",        1.50,   0.00,    0),
    ("CAM_FRONT_LEFT",   1.00,   0.80,  -55),
    ("CAM_FRONT_RIGHT",  1.00,  -0.80,   55),
    ("CAM_BACK",        -1.80,   0.00,  180),
    ("CAM_BACK_LEFT",   -1.00,   0.80, -125),
    ("CAM_BACK_RIGHT",  -1.00,  -0.80,  125),
    ("LIDAR",            0.00,   0.00,    0),
    ("RAD_FRONT",        2.10,   0.00,    0),
    ("RAD_LEFT",         0.00,   0.80,  -90),
    ("RAD_RIGHT",        0.00,  -0.80,   90),
    ("RAD_BACK",        -2.10,   0.00,  180),
]

ARROW_LEN = 0.32
MR        = 0.065
CX        = '#CC1111'
CY        = '#119911'

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_aspect('equal')
ax.axis('off')

# ── Car body: straight door sides + angular corners ────────────────────────
# Half-outline (top, y≥0), front→rear, y=0 at both tips.
# Door section (x: 1.25 → -1.45) has nearly constant y=0.96–1.00 → STRAIGHT.
tx = np.array([
    # Front bumper + hood (right side)
    2.42, 2.36, 2.20, 1.98, 1.72, 1.48, 1.28, 1.10,
    # Door section — nearly flat, just 5 pts
    0.70, 0.00, -0.70, -1.18,
    # Rear quarter + trunk
    -1.40, -1.60, -1.82, -2.05, -2.22, -2.36, -2.42
])
ty = np.array([
    # bumper/hood ramps up steeply then more gradually
    0.00, 0.15, 0.38, 0.60, 0.76, 0.86, 0.93, 0.97,
    # door — essentially a straight line at y≈0.99–1.00
    0.99, 1.00, 0.99, 0.97,
    # rear quarter/trunk tapers back down
    0.93, 0.87, 0.75, 0.58, 0.40, 0.18, 0.00
])

bx = np.concatenate([tx, tx[-2:0:-1]])
by = np.concatenate([ty, -ty[-2:0:-1]])

ax.fill(bx, by, color='#0c0c20', zorder=1, linewidth=0)
ax.plot(np.append(bx, bx[0]), np.append(by, by[0]),
        color='#28285a', lw=0.9, zorder=2)

# ── Roof: clearly rectangular central panel ────────────────────────────────
# Straight sides from x=0.92 to x=-1.32 keep y≈0.63 → car-like roof box
roof_top_x = np.array([1.10, 1.05, 0.92, -1.32, -1.44, -1.64, -1.76])
roof_top_y = np.array([0.00, 0.32, 0.63,  0.63,  0.56,  0.38,  0.00])
rox = np.concatenate([roof_top_x, roof_top_x[-2:0:-1]])
roy = np.concatenate([roof_top_y, -roof_top_y[-2:0:-1]])
ax.fill(rox, roy, color='#161638', zorder=3, linewidth=0)

# ── Sensors ────────────────────────────────────────────────────────────────
AKW = dict(lw=1.0, mutation_scale=7, zorder=12)

for name, fx, fy, yaw in SENSORS:
    sx, sy = fx, fy
    Xv, Yv = sensor_axes(yaw)

    ax.add_patch(Circle((sx, sy), MR, color='white', fill=False,
                         linewidth=0.9, zorder=11))
    ax.plot([sx-MR, sx+MR], [sy, sy],   color='white', lw=0.55, zorder=11)
    ax.plot([sx, sx],   [sy-MR, sy+MR], color='white', lw=0.55, zorder=11)

    ax.annotate('', xy=(sx + Xv[0]*ARROW_LEN, sy + Xv[1]*ARROW_LEN),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=CX, **AKW))
    ax.annotate('', xy=(sx + Yv[0]*ARROW_LEN, sy + Yv[1]*ARROW_LEN),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=CY, **AKW))

# ── Labels (all with correct placement inside/outside body) ───────────────
# Corner cameras offset INWARD (toward y=0) to stay on dark body
LPOS = {
    "CAM_FRONT":       ( 0.00,  0.30, 'center', 'bottom', 'white', False),
    "CAM_FRONT_LEFT":  ( 0.00, -0.24, 'center', 'top',    'white', False),
    "CAM_FRONT_RIGHT": ( 0.00,  0.24, 'center', 'bottom', 'white', False),
    "CAM_BACK":        ( 0.00,  0.28, 'center', 'bottom', 'white', False),
    "CAM_BACK_LEFT":   ( 0.00, -0.24, 'center', 'top',    'white', False),
    "CAM_BACK_RIGHT":  ( 0.00,  0.24, 'center', 'bottom', 'white', False),
    "LIDAR":           ( 0.00, -0.26, 'center', 'top',    'white', False),
    "RAD_FRONT":       ( 0.80,  0.00, 'left',   'center', 'black', True),
    "RAD_LEFT":        ( 0.00,  0.76, 'center', 'bottom', 'black', True),
    "RAD_RIGHT":       ( 0.00, -0.76, 'center', 'top',    'black', True),
    "RAD_BACK":        (-0.80,  0.00, 'right',  'center', 'black', True),
}

for name, fx, fy, yaw in SENSORS:
    sx, sy = fx, fy
    dx, dy, ha, va, tc, line = LPOS[name]
    lx, ly = sx + dx, sy + dy
    if line:
        ax.plot([sx, lx], [sy, ly], color='#999999', lw=0.6, zorder=10)
    ax.text(lx, ly, name, ha=ha, va=va, fontsize=6.5,
            color=tc, fontfamily='monospace', fontweight='bold', zorder=13)

# ── Legend ─────────────────────────────────────────────────────────────────
ax.legend(handles=[
    Line2D([0],[0], color=CX, lw=1.6, label='X'),
    Line2D([0],[0], color=CY, lw=1.6, label='Y'),
], loc='lower right', fontsize=8.5, framealpha=0.92,
   edgecolor='#bbbbbb', handlelength=1.5, handletextpad=0.5, borderpad=0.6)

ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-1.95, 1.95)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, bbox_inches='tight', dpi=200)
print(f"Wrote {OUT}")
