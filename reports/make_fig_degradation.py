import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
TXT, MUTED, GRID, SURF = "#22221f", "#63635e", "#dedbd3", "#fcfcfb"
EPS = [0.0, 0.25, 0.50, 0.75, 1.00]
#            carbon                                         completion                       chi
blend_c = [0.186251, 0.154356, 0.142712, 0.138325, 0.126223]
blend_p = [100.00, 99.91, 97.19, 97.74, 93.58]
blend_x = [0.7085, 0.7094, 0.7090, 0.7085, 0.0000]
shuf_c  = [0.186251, 0.196450, 0.225239, 0.220580, 0.227747]
shuf_p  = [100.00, 93.82, 100.00, 100.00, 100.00]
shuf_x  = [0.7085, 0.4711, 0.2300, 0.2307, 0.2305]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
fig.patch.set_facecolor(SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xticks(EPS); ax.set_xlim(-0.06, 1.06)
    ax.set_xlabel(r"corruption strength $\varepsilon$", color=TXT, fontsize=8.5)

def draw(ax, y, comp, col, lw=2):
    ax.plot(EPS, y, color=col, lw=lw, zorder=3)
    for e, v, c in zip(EPS, y, comp):
        ok = c >= 99.5
        ax.plot(e, v, "o", ms=6.5, zorder=4,
                color=col if ok else SURF, mec=col, mew=1.8)

a = axes[0]
draw(a, shuf_c, shuf_p, ORANGE); draw(a, blend_c, blend_p, BLUE)
a.axhline(blend_c[0], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
a.set_ylabel("carbon per completed work", color=TXT, fontsize=8.5)
a.set_ylim(0.118, 0.245)
a.text(1.0, 0.2335, "coherent lie", color=ORANGE, fontsize=8, ha="right")
a.text(0.52, 0.1245, "information removal", color=BLUE, fontsize=8, ha="center")
a.text(0.30, 0.1885, "clean", color=MUTED, fontsize=7.5, ha="left")
a.plot([], [], "o", ms=6.5, color=SURF, mec=MUTED, mew=1.8,
       label="completion < 99.5%")
a.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=MUTED,
         handletextpad=0.35, bbox_to_anchor=(-0.03, 1.04))

b = axes[1]
draw(b, shuf_x, shuf_p, ORANGE); draw(b, blend_x, blend_p, BLUE)
b.axhline(0.2, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
b.text(0.02, 0.225, "gate threshold", color=MUTED, fontsize=7)
b.set_ylabel(r"auditor statistic $\chi$", color=TXT, fontsize=8.5)
b.set_ylim(-0.05, 0.83)
b.text(1.0, 0.30, "coherent lie", color=ORANGE, fontsize=8, ha="right")
b.text(0.50, 0.755, "information removal", color=BLUE, fontsize=8, ha="center")
b.annotate("blind until total", xy=(0.50, 0.709), xytext=(0.42, 0.50),
           color=BLUE, fontsize=7, ha="center",
           arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
fig.tight_layout(pad=0.5, w_pad=2.0)
fig.savefig("paper_latest/figs/fig_degradation.png", dpi=320)
print("written")
