#!/usr/bin/env python3
"""Live monitor for lbm.log — run from the simulation working directory."""

import re
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

LOG_FILE = "lbm.log"
POLL_MS = 2000  # refresh interval in milliseconds

# ── Parsing ──────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^--- (.+) ---$")
_KV_RE = re.compile(r"^\s{2}(\S.*?)\s*=\s*(.+)$")
_DATA_RE = re.compile(
    r"Time (\d+): Mass: ([^\s,]+), "
    r"Px: ([^\s,]+), Py: ([^\s,]+), Pz: ([^\s,]+), "
    r"Kinetic Energy: ([^\s]+) Relative Error: ([^\s]+)"
)


def _safe_float(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_log(path):
    """Return (params: dict-of-sections, rows: list-of-tuples)."""
    params = {}
    current_section = None
    rows = []

    try:
        with open(path, "r") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}, []

    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current_section = m.group(1)
            params[current_section] = {}
            continue

        if current_section:
            m = _KV_RE.match(line)
            if m:
                params[current_section][m.group(1).strip()] = m.group(2).strip()
                continue

        m = _DATA_RE.match(line)
        if m:
            rows.append(tuple(_safe_float(g) for g in m.groups()))

    return params, rows


def format_params(params):
    lines = []
    for section, kvs in params.items():
        lines.append(f"─── {section}")
        for k, v in kvs.items():
            # Wrap long lines
            entry = f"  {k} = {v}"
            lines.append(entry)
    return "\n".join(lines)


# ── Styling ───────────────────────────────────────────────────────────────────

BG = "#1e1e2e"
PANEL = "#181825"
FG = "#cdd6f4"
SUBTEXT = "#a6adc8"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
RED = "#f38ba8"
BLUE = "#89b4fa"
MAUVE = "#cba6f7"
PEACH = "#fab387"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.title.set_color(FG)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = os.path.join(os.getcwd(), LOG_FILE)
    print(f"Monitoring {log_path}  (refresh every {POLL_MS / 1000:.1f} s)")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.canvas.manager.set_window_title("LBM Monitor")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[2.6, 1],
        height_ratios=[1.3, 1],
        hspace=0.42, wspace=0.28,
        left=0.07, right=0.97, top=0.91, bottom=0.09,
    )

    ax_ke = fig.add_subplot(gs[0, 0])
    ax_mom = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    style_ax(ax_ke)
    style_ax(ax_mom)

    ax_info.set_facecolor(PANEL)
    ax_info.axis("off")
    for spine in ax_info.spines.values():
        spine.set_color("#45475a")

    # KE axes
    (ke_line,) = ax_ke.plot([], [], color=BLUE, lw=1.5, label="KE")
    ax_ke.set_xlabel("Time step")
    ax_ke.set_ylabel("Kinetic Energy")
    ax_ke.set_title("Kinetic Energy")
    ax_ke.set_yscale("log")

    # Momentum axes
    (px_line,) = ax_mom.plot([], [], color=MAUVE, lw=1.2, label="Px")
    (py_line,) = ax_mom.plot([], [], color=PEACH, lw=1.2, label="Py", alpha=0.8)
    (pz_line,) = ax_mom.plot([], [], color=GREEN, lw=1.2, label="Pz", alpha=0.7)
    ax_mom.set_xlabel("Time step")
    ax_mom.set_ylabel("Momentum")
    ax_mom.set_title("Net Momentum")
    leg = ax_mom.legend(fontsize=7, facecolor=PANEL, edgecolor="#45475a",
                        labelcolor=FG, loc="upper left")

    # Info panel: parameters (top) + mass status (bottom)
    param_text = ax_info.text(
        0.04, 0.98, "Loading…",
        transform=ax_info.transAxes,
        fontsize=7.5, va="top", ha="left",
        family="monospace", color=FG,
        wrap=False,
    )

    # Mass status box — sits just below the params text
    mass_box = ax_info.text(
        0.04, 0.05, "",
        transform=ax_info.transAxes,
        fontsize=9, va="bottom", ha="left",
        family="monospace", weight="bold", color=GREEN,
    )

    # Title bar
    title_obj = fig.suptitle(
        "LBM Monitor — waiting for data…",
        color=FG, fontsize=11, y=0.97,
    )

    # ── Persistent state ─────────────────────────────────────────────────────
    state = {"mass0": None, "params_set": False}

    def update(_frame):
        params, rows = parse_log(log_path)

        if not rows:
            title_obj.set_text(f"LBM Monitor — waiting for {LOG_FILE}…")
            return

        # Set params text once (they don't change mid-run)
        if not state["params_set"] and params:
            param_text.set_text(format_params(params))
            state["params_set"] = True

        # Unpack columns
        arr = np.array(rows)
        times = arr[:, 0]
        masses = arr[:, 1]
        px, py, pz = arr[:, 2], arr[:, 3], arr[:, 4]
        kes = arr[:, 5]

        if state["mass0"] is None:
            state["mass0"] = masses[0]
        mass0 = state["mass0"]

        # ── KE plot ──────────────────────────────────────────────────────────
        valid = np.isfinite(kes) & (kes > 0)
        if valid.any():
            ke_line.set_data(times[valid], kes[valid])
            ax_ke.relim()
            ax_ke.autoscale_view()

        # ── Momentum plot ─────────────────────────────────────────────────────
        px_line.set_data(times, px)
        py_line.set_data(times, py)
        pz_line.set_data(times, pz)
        ax_mom.relim()
        ax_mom.autoscale_view()

        # ── Mass status ───────────────────────────────────────────────────────
        rel_err = abs(masses[-1] - mass0) / abs(mass0) if mass0 != 0 else 0.0
        if rel_err < 1e-8:
            color, label = GREEN, "MASS  ✓"
        elif rel_err < 1e-5:
            color, label = YELLOW, "MASS  ⚠"
        else:
            color, label = RED, "MASS  ✗"

        mass_box.set_color(color)
        mass_box.set_text(
            f"{label}\n"
            f"|ΔM/M₀| = {rel_err:.2e}\n"
            f"M₀ = {mass0:.6g}\n"
            f"M  = {masses[-1]:.9g}"
        )

        # ── Title ─────────────────────────────────────────────────────────────
        t_cur = int(times[-1])
        ke_cur = kes[-1]
        title_obj.set_text(
            f"LBM Monitor  │  t = {t_cur:,}  │  KE = {ke_cur:.4e}  │  steps = {len(rows)}"
        )

    ani = animation.FuncAnimation(
        fig, update, interval=POLL_MS, cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()
