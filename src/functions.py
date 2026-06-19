"""
Funktionen für die Analyse von Wortassoziationen in Wikipedia-Artikeln über Wissenschaftler:innen, aufgeteilt nach Geschlecht. Enthält Funktionen zur Berechnung der Pointwise Mutual Information (PMI) und
zur Visualisierung der Ergebnisse.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wikipediaapi






# Visualisierungen
# Balkendiagramm der Geschlechterverteilung

def plot_gender(df):
    counts = df.groupby("genderLabel").size()

    fig, ax = plt.subplots(figsize=(8, 5))

    blues = ["#85B7EB", "#042C53"]
    colors = [blues[i % len(blues)] for i in range(len(counts))]

    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#042C53",
                  linewidth=1.2, width=0.55)

    # Anzahl der Geschlechter oberhalb des Balkens
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + counts.values.max() * 0.01,
                f"{int(height):,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#0C447C")

    # Styling
    ax.set_facecolor("#fcfbf9")
    fig.patch.set_facecolor("#fcfbf9")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B5D4F4")
    ax.tick_params(axis="x", colors="#185FA5", labelsize=11)
    ax.tick_params(axis="y", colors="#378ADD", labelsize=10)
    ax.grid(axis="y", color="#B5D4F4", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Verteilung Geschlechter", fontsize=20, fontweight="bold",
                 color="#042C53", pad=15)
    plt.tight_layout()
    plt.savefig("../figures/verteilung_geschlechter.png", dpi=300)
    plt.show()

# Plot zur Pointwise Mutual Information (PMI)
def plot_pmi(pmi_data: dict, top_n: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#fcfbf9")

    COLORS = {"männlich": "#85B7EB", "weiblich": "#042C53"}

    for ax, gender in zip(axes, ("männlich", "weiblich")):
        words = list(pmi_data[gender].keys())[:top_n]
        scores = [pmi_data[gender][w] for w in words]
        y_pos = np.arange(len(words))

        ax.barh(y_pos, scores, color=COLORS[gender], alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10, color="#0C447C")
        ax.invert_yaxis()

        ax.set_facecolor("#fcfbf9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#B5D4F4")
        ax.tick_params(axis="x", colors="#378ADD", labelsize=9)
        ax.tick_params(axis="y", colors="#0C447C", labelsize=12)
        ax.grid(axis="x", color="#B5D4F4", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("PMI (Bits)")
        ax.set_title(f"Top-Wörter assoziiert mit {'männlichen' if gender == 'männlich' else 'weiblichen'} Wissenschaftler:innen")
        ax.axvline(0, color="#185FA5", linewidth=0.9, linestyle="-")
    x_min = min(ax.get_xlim()[0] for ax in axes)
    x_max = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    plt.suptitle("Wortassoziationen nach Geschlecht (PMI)",
                 fontsize=20, fontweight="bold", color="#042C53", y=1.01)
    plt.tight_layout()
    plt.show()

# Scatterplot zur PMI
def plot_rank_scatter(freq_m: dict, freq_f: dict):
    common = set(freq_m) & set(freq_f)
    words = list(common)
    fm = np.array([freq_m[w] for w in words])
    ff = np.array([freq_f[w] for w in words])

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#fcfbf9")
    ax.set_facecolor("#fcfbf9")

    ax.scatter(np.log10(fm), np.log10(ff), alpha=0.25, s=6, color="#85B7EB")

    # Interessanteste Wörter annotieren
    interesting = sorted(common, key=lambda w: abs(freq_m[w] - freq_f[w]),
                         reverse=True)[:12]
    for w in interesting:
        ax.annotate(w, (np.log10(freq_m[w]), np.log10(freq_f[w])),
                    fontsize=10, color="#042C53", alpha=0.9,
                    xytext=(4, 4), textcoords="offset points")

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color="#378ADD", linewidth=0.9,
            linestyle="--", label="y = x")

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B5D4F4")
    ax.tick_params(axis="both", colors="#378ADD", labelsize=9)
    ax.grid(color="#B5D4F4", linestyle="--", linewidth=0.6, alpha=0.7)

    ax.set_xlabel("log₁₀(Häufigkeit) — männliche Wissenschaftler",
                  fontsize=11, color="#185FA5", labelpad=8)
    ax.set_ylabel("log₁₀(Häufigkeit) — weibliche Wissenschaftler",
                  fontsize=11, color="#185FA5", labelpad=8)
    ax.set_title("Worthäufigkeits-Scatter (gemeinsamer Wortschatz)",
                 fontsize=13, fontweight="bold", color="#042C53", pad=12)
    rho, p = spearmanr(fm, ff)
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f}",
            transform=ax.transAxes, color="#042C53", fontsize=10)
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor("#fcfbf9")
    legend.get_frame().set_edgecolor("#B5D4F4")
    for text in legend.get_texts():
        text.set_color("#185FA5")

    plt.tight_layout()
    plt.show()