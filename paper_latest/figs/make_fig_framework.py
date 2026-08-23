"""Main-text EU-CRD schematic.

The drawio original packed seven numbered blocks, full equations and an eight-entry
arrow legend into one panel; reviewers called it unreadable, and its forecast-
responsibility box still showed the unnormalised form that Eq. 2 no longer uses.
This redraw keeps one argument only: a transition's credit is split between the
forecast channel and the policy channel, and only the policy share reaches PPO.
Two accent colours carry that split, blue for the forecast, orange for the policy.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE = "#2f6fa8"
BLUE_FILL = "#dce9f4"
ORANGE = "#c2681f"
ORANGE_FILL = "#f7e5d2"
GREY = "#4a4a4a"
GREY_FILL = "#ececec"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.2,
})

FIG_W, FIG_H = 7.0, 3.05
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, edge, fill, fs=8.2, weight="normal", lw=1.0, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#111111", zorder=3, fontweight=weight,
            linespacing=1.45)
    return (x, y, w, h)


def arrow(p0, p1, color=GREY, ls="-", lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
        color=color, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


# ---------------------------------------------------------------- training lane
TOP, BOT = 78, 53          # centre lines of the two counterfactual rows
H = 17                      # standard box height

trans = box(1.5, 65.5, 14, H,
            "Transition\n$(s_t,\\, a_t,\\, r^{G}_t)$", GREY, GREY_FILL)

fc = box(20, TOP - H / 2, 22, H,
         "Forecast counterfactual\nrealised $g$ vs forecast $\\hat{g}$\n$\\Rightarrow R_{\\mathrm{forecast}}$",
         BLUE, BLUE_FILL, fs=7.8)

rc = box(20, BOT - H / 2, 22, H,
         "Routing counterfactual\n$\\tilde{a}_t \\sim \\pi_G(\\cdot\\,|\\,s_t)$\n$\\Rightarrow R_{\\mathrm{global}}$",
         ORANGE, ORANGE_FILL, fs=7.8)

gate = box(18.5, BOT - H / 2 - 12.5, 25, 10,
           "gate  $c_t=\\exp(-\\sigma^{2}_t/\\tau_t)$\n$R_{\\mathrm{global}}=c_t\\,\\Delta Q+(1-c_t)\\,\\Delta r$",
           ORANGE, "white", fs=7.2, ls=(0, (3, 2)))

share = box(46.0, 56.0, 20, 29,
            "Responsibility shares\n$\\rho_{\\mathrm{forecast}},\\ \\rho_{\\mathrm{global}}$\nscale-normalised and\nfloored at $\\rho_{\\min}$",
            GREY, "white", fs=7.4)

rw = box(69, 65.5, 19, H,
         "Reweighted advantage\n$\\tilde{A}_t=(\\rho_{\\mathrm{global}}/\\bar{\\rho})\\,\\hat{A}_t$",
         ORANGE, ORANGE_FILL, fs=7.8)

ppo = box(91, 65.5, 7.5, H, "PPO\nupdate\n$\\theta$", GREY, GREY_FILL, fs=7.8)

arrow(right(trans), (fc[0], TOP - 2.5), rad=-0.12)
arrow(right(trans), (rc[0], BOT + 2.5), rad=0.12)
arrow((fc[0] + fc[2], TOP), (share[0], TOP - 2), color=BLUE, rad=0.08)
arrow((rc[0] + rc[2], BOT), (share[0], BOT + 5), color=ORANGE, rad=-0.08)
arrow((gate[0] + gate[2], BOT - H / 2 - 7.5), (rc[0] + rc[2] + 1.2, BOT - 4),
      color=ORANGE, ls=(0, (3, 2)), lw=0.9, rad=-0.25)
arrow(right(share), left(rw), color=ORANGE)
arrow(right(rw), left(ppo))

# quarantine: the forecast share is computed and then withheld from the gradient
ax.plot([56, 56], [85.0, 90.5], color=BLUE, lw=1.1, ls=(0, (3, 2)), zorder=4)
ax.plot([54.3, 57.7], [88.8, 92.2], color=BLUE, lw=1.4, zorder=5)
ax.plot([54.3, 57.7], [92.2, 88.8], color=BLUE, lw=1.4, zorder=5)
ax.text(59.5, 90.5, "$\\rho_{\\mathrm{forecast}}$ quarantined:\nno gradient reaches the policy",
        ha="left", va="center", fontsize=7.2, color=BLUE, linespacing=1.4)

ax.text(1.5, 94.0, "Training", fontsize=8.6, fontweight="bold", color="#111111")

# ------------------------------------------------------------- deployment lane
ax.add_patch(Rectangle((1.5, 2.5), 97, 25, linewidth=1.0, edgecolor="#8a8a8a",
                       facecolor="#f7f7f7", linestyle=(0, (4, 2.5)), zorder=1))
ax.text(3.2, 23.7, "Deployment", fontsize=8.6, fontweight="bold", color="#111111")

aud = box(15.5, 5.5, 22, 14.5,
          "Auditor\n$\\chi_i=\\mathrm{corr}(\\hat{y}^{s}_i,\\, g_i)$  per DC",
          BLUE, BLUE_FILL, fs=7.8)
resp = box(43.5, 5.5, 27, 14.5,
           "$\\chi_i\\!\\to\\!0$: suppress DEFER\n$\\chi_i\\!\\ll\\!0$: invert that DC's\nforecast features",
           BLUE, "white", fs=7.2)
back = box(76.5, 5.5, 18, 14.5, "Router $\\pi_G$\n(unchanged weights)", GREY, GREY_FILL, fs=7.8)
arrow(right(aud), left(resp), color=BLUE)
arrow(right(resp), left(back), color=BLUE)
ax.text(3.2, 12.0, "no\nretraining", ha="left", va="center", fontsize=7.0, color="#555555")

fig.savefig("/home/joshua/rl-cloudsimplus-greenscheduling/paper_latest/figs/fig_framework_main.pdf")
fig.savefig("/home/joshua/rl-cloudsimplus-greenscheduling/paper_latest/figs/fig_framework_main.png", dpi=260)
print("written")
