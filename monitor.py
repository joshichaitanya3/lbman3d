#!/usr/bin/env python3
"""Live monitor for lbm.log — run from the simulation working directory."""

import argparse
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
    r"Kinetic Energy: ([^\s,]+)"
    r"(?:,\s*Total Energy: ([^\s,]+))?"   # optional; absent in older logs
    r"[, ]+Relative Error: ([^\s,]+)"      # comma (new) or space (old) separator
    r"(?:,\s*NumDisclinations: (\d+))?"
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
            groups = m.groups()
            # groups: time, mass, px, py, pz, ke, total_e?, rel_err, disc?
            row = tuple(_safe_float(g) for g in groups[:6])   # time..ke
            total_e = _safe_float(groups[6]) if groups[6] is not None else float("nan")
            rel_err = _safe_float(groups[7])
            disc = _safe_float(groups[8]) if groups[8] is not None else float("nan")
            # tuple layout: time, mass, px, py, pz, ke, rel_err, disc, total_e
            rows.append(row + (rel_err, disc, total_e))

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
    ap = argparse.ArgumentParser(description="Live monitor for lbm.log")
    ap.add_argument(
        "--total_energy", action="store_true",
        help="Plot Total Energy instead of Kinetic Energy (useful for passive benchmarks)",
    )
    args = ap.parse_args()
    plot_total_energy = args.total_energy

    log_path = os.path.join(os.getcwd(), LOG_FILE)
    print(f"Monitoring {log_path}  (refresh every {POLL_MS / 1000:.1f} s)")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.canvas.manager.set_window_title("LBM Monitor")

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[2.6, 1],
        height_ratios=[1.3, 1, 1],
        hspace=0.52, wspace=0.28,
        left=0.07, right=0.97, top=0.91, bottom=0.09,
    )

    ax_ke = fig.add_subplot(gs[0, 0])
    ax_mom = fig.add_subplot(gs[1, 0])
    ax_disc = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    style_ax(ax_ke)
    style_ax(ax_mom)
    style_ax(ax_disc)

    ax_info.set_facecolor(PANEL)
    ax_info.axis("off")
    for spine in ax_info.spines.values():
        spine.set_color("#45475a")

    # Top energy axes — KE by default, Total Energy with --total_energy
    if plot_total_energy:
        top_label, top_col, top_color = "Total Energy", 8, PEACH
    else:
        top_label, top_col, top_color = "Kinetic Energy", 5, BLUE

    (ke_line,) = ax_ke.plot([], [], color=top_color, lw=1.5)
    ax_ke.set_xlabel("Time step")
    ax_ke.set_ylabel(top_label)
    ax_ke.set_title(top_label)
    if not plot_total_energy:
        ax_ke.set_yscale("log")  # KE is always positive; Total Energy can be negative

    # Momentum axes
    (px_line,) = ax_mom.plot([], [], color=MAUVE, lw=1.2, label="Px")
    (py_line,) = ax_mom.plot([], [], color=PEACH, lw=1.2, label="Py", alpha=0.8)
    (pz_line,) = ax_mom.plot([], [], color=GREEN, lw=1.2, label="Pz", alpha=0.7)
    ax_mom.set_xlabel("Time step")
    ax_mom.set_ylabel("Momentum")
    ax_mom.set_title("Net Momentum")
    leg = ax_mom.legend(fontsize=7, facecolor=PANEL, edgecolor="#45475a",
                        labelcolor=FG, loc="upper left")

    # Disclinations axes
    (disc_line,) = ax_disc.plot([], [], color=RED, lw=1.5)
    ax_disc.set_xlabel("Time step")
    ax_disc.set_ylabel("Count")
    ax_disc.set_title("Disclinations")

    # Info panel: parameters (top) + mass status (bottom)
    param_text = ax_info.text(
        0.04, 0.98, "Loading…",
        transform=ax_info.transAxes,
        fontsize=7.5, va="top", ha="left",
        family="monospace", color=FG,
        wrap=False,
    )

    # Mass status box — sits at the very bottom of the info panel
    mass_box = ax_info.text(
        0.04, 0.05, "",
        transform=ax_info.transAxes,
        fontsize=9, va="bottom", ha="left",
        family="monospace", weight="bold", color=GREEN,
    )

    # Energy trend indicator — only created when --total_energy is active
    if plot_total_energy:
        energy_box = ax_info.text(
            0.04, 0.22, "",
            transform=ax_info.transAxes,
            fontsize=9, va="bottom", ha="left",
            family="monospace", weight="bold", color=SUBTEXT,
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
        discs = arr[:, 7]

        if state["mass0"] is None:
            state["mass0"] = masses[0]
        mass0 = state["mass0"]

        # ── Top energy plot (KE or Total Energy) ─────────────────────────────
        top_vals = arr[:, top_col]
        valid = np.isfinite(top_vals) & (top_vals != 0 if plot_total_energy else top_vals > 0)
        if valid.any():
            ke_line.set_data(times[valid], top_vals[valid])
            ax_ke.relim()
            ax_ke.autoscale_view()

        # ── Momentum plot ─────────────────────────────────────────────────────
        px_line.set_data(times, px)
        py_line.set_data(times, py)
        pz_line.set_data(times, pz)
        ax_mom.relim()
        ax_mom.autoscale_view()

        # ── Disclinations plot ────────────────────────────────────────────────
        valid_disc = np.isfinite(discs)
        if valid_disc.any():
            disc_line.set_data(times[valid_disc], discs[valid_disc])
            ax_disc.relim()
            ax_disc.autoscale_view()

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

        # ── Energy trend indicator (--total_energy mode only) ────────────────
        if plot_total_energy:
            te_vals = arr[:, 8]
            finite_te = te_vals[np.isfinite(te_vals)]
            if len(finite_te) >= 2:
                delta = finite_te[-1] - finite_te[-2]
                if delta < 0:
                    e_color, e_label = GREEN, "ENERGY  ↓"
                else:
                    e_color, e_label = RED, "ENERGY  ↑"
                energy_box.set_color(e_color)
                energy_box.set_text(f"{e_label}\ndE = {delta:.3e}")

        # ── Title ─────────────────────────────────────────────────────────────
        t_cur = int(times[-1])
        top_cur = arr[-1, top_col]
        title_obj.set_text(
            f"LBM Monitor  │  t = {t_cur:,}  │  {top_label} = {top_cur:.4e}  │  steps = {len(rows)}"
        )

    ani = animation.FuncAnimation(
        fig, update, interval=POLL_MS, cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()
