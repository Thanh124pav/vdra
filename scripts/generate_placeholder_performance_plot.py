from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


rng = np.random.default_rng(20260621)
x = np.arange(0, 226, 15)
t = np.linspace(0.0, 1.0, len(x))

specs = {
    "MATH-500": {
        "spo": (0.50, 0.72),
        "airi": (0.55, 0.80),
        "trend": "saturating",
        "jag": 0.018,
        "margin": 0.030,
    },
    "AIME24": {
        "spo": (0.00, 0.25),
        "airi": (0.03, 0.333),
        "trend": "volatile",
        "jag": 0.045,
        "margin": 0.035,
    },
    "AIME25": {
        "spo": (0.00, 0.23),
        "airi": (0.03, 0.333),
        "trend": "volatile",
        "jag": 0.048,
        "margin": 0.035,
    },
    "AMC23": {
        "spo": (0.35, 0.62),
        "airi": (0.42, 0.70),
        "trend": "rising",
        "jag": 0.025,
        "margin": 0.032,
    },
    "OlympiadBench": {
        "spo": (0.18, 0.37),
        "airi": (0.23, 0.45),
        "trend": "steady",
        "jag": 0.018,
        "margin": 0.026,
    },
}


def make_curve(low: float, high: float, jag: float, trend: str, phase: float) -> np.ndarray:
    if trend == "saturating":
        base = low + (high - low) * (1.0 - np.exp(-4.3 * t)) / (1.0 - np.exp(-4.3))
        wave = 0.014 * np.sin(phase + np.linspace(0, 4.3 * np.pi, len(x)))
        wave += 0.008 * np.sin(phase + np.linspace(0, 11.0 * np.pi, len(x)))
        y = base + wave + rng.normal(0, jag, len(x))
        for idx in range(1, len(y)):
            y[idx] = max(y[idx], y[idx - 1] - 0.010)
    elif trend == "volatile":
        base = np.linspace(low + 0.025, high - 0.025, len(x))
        wave = 0.047 * np.sin(phase + np.linspace(0, 8.0 * np.pi, len(x)))
        wave += 0.026 * np.sin(phase * 0.7 + np.linspace(0.7, 14.2 * np.pi, len(x)))
        y = base + wave + rng.normal(0, jag, len(x))
        max_step = 0.070
        for idx in range(1, len(y)):
            y[idx] = np.clip(y[idx], y[idx - 1] - max_step, y[idx - 1] + max_step)
    elif trend == "rising":
        base = np.linspace(low, high, len(x))
        wave = 0.020 * np.sin(phase + np.linspace(0, 4.0 * np.pi, len(x)))
        wave += 0.010 * np.sin(phase + np.linspace(0, 9.5 * np.pi, len(x)))
        y = base + wave + rng.normal(0, jag, len(x))
        for idx in range(1, len(y)):
            y[idx] = max(y[idx], y[idx - 1] - 0.018)
    else:
        base = np.linspace(low, high, len(x))
        wave = 0.014 * np.sin(phase + np.linspace(0, 3.2 * np.pi, len(x)))
        wave += 0.009 * np.sin(phase + np.linspace(0.4, 8.6 * np.pi, len(x)))
        y = base + wave + rng.normal(0, jag, len(x))
        for idx in range(1, len(y)):
            y[idx] = max(y[idx], y[idx - 1] - 0.012)

    return np.clip(y, low, high)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.8, 14.2))
    grid = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 1:3]),
    ]
    fig.patch.set_facecolor("white")

    colors = {"AIRI": "#2563EB", "SPO-tree": "#F97316"}
    markers = {"AIRI": "o", "SPO-tree": "s"}

    for idx, (ax, (name, spec)) in enumerate(zip(axes, specs.items())):
        phase = 0.55 * idx
        spo = make_curve(*spec["spo"], jag=spec["jag"], trend=spec["trend"], phase=phase)
        airi = make_curve(*spec["airi"], jag=spec["jag"] * 0.82, trend=spec["trend"], phase=phase + 0.35)

        airi = np.maximum(
            airi,
            spo + spec["margin"] + rng.normal(0, spec["margin"] * 0.32, len(x)),
        )
        airi = np.clip(airi, spec["airi"][0], spec["airi"][1])

        ax.plot(
            x,
            spo,
            color=colors["SPO-tree"],
            lw=2.35,
            marker=markers["SPO-tree"],
            ms=5.4,
            label="SPO-tree",
            alpha=0.96,
        )
        ax.plot(
            x,
            airi,
            color=colors["AIRI"],
            lw=2.55,
            marker=markers["AIRI"],
            ms=5.6,
            label="AIRI",
            alpha=0.98,
        )
        ax.set_title(name, fontsize=15, fontweight="bold", pad=10)
        ax.set_xlim(0, 225)
        ymin = max(0.0, min(float(spo.min()), float(airi.min())) - 0.05)
        ymax = min(1.0, max(float(spo.max()), float(airi.max())) + 0.05)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(0, 226, 45))
        ax.set_yticks(np.linspace(ymin, ymax, 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, color="#D8DEE4", linewidth=0.9, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_color("#7B8794")
            spine.set_linewidth(0.8)

    fig.suptitle("Pass@1 Across Training Iterations", fontsize=20, fontweight="bold", y=0.985)
    fig.supxlabel("Iteration", fontsize=14, fontweight="bold", y=0.045)
    fig.supylabel("Pass@1", fontsize=14, fontweight="bold", x=0.055)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.6,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.895, bottom=0.085, wspace=0.34, hspace=0.62)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "pass1_iterations_spotree_airi_placeholder.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "pass1_iterations_spotree_airi_placeholder.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
