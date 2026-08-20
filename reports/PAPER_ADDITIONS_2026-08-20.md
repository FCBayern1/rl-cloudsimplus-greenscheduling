% =====================================================================
% Paper additions, 2026-08-20 (5080). Four self-contained blocks.
% Every number below is measured; provenance is in the comment above it.
% Paste points are marked. Nothing here rewrites existing text.
% =====================================================================


% ---------------------------------------------------------------------
% BLOCK 1 -- degradation figure.  Paste into Section 4 (sec:main),
% after the Table 1 discussion.  Needs figs/fig_degradation.png.
% Data: local_eval_rt/corruption_sweep_van.txt (Vanilla, seed 3, 5
% episodes per cell, argmax decoding, 9 cells).
% ---------------------------------------------------------------------
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{./figs/fig_degradation.png}
    \caption{Degradation under graded corruption, Vanilla PPO (one seed, five
    episodes per cell). Both operators reduce to the Table~\ref{tab:main}
    conditions at $\varepsilon{=}1$. Left: a coherent lie raises carbon
    monotonically while completion holds, so the loss is not bought by
    dropping work; removing information instead lowers carbon and completion
    together, the deflation the iso-completion rule exists to catch (hollow
    markers, not comparable). Right: the auditor statistic tracks the lie but
    is blind to partial removal --- blending a forecast towards a constant is
    an affine shrink, and correlation is invariant to it, so $\chi$ moves only
    when the last of the variance is gone.}
    \label{fig:degradation}
\end{figure}


% ---------------------------------------------------------------------
% BLOCK 2 -- paragraph for Section 4, right after the figure.
% ---------------------------------------------------------------------
Table~\ref{tab:main} reports two corruption points; Figure~\ref{fig:degradation}
fills in the path between them. Carbon rises monotonically with the coherence of
the lie, from $0.186$ clean to $0.225$ at half-permuted and $0.228$ at full,
with completion pinned at 100\% throughout, so the penalty is paid in carbon
rather than in dropped work. The information-removal axis behaves in the
opposite and more instructive way: carbon falls to $0.126$, but completion falls
with it to 93.6\%, and only the mildest cell stays inside the contract. Read
together the two axes separate two failure modes that a single corruption point
conflates --- a forecast that lies coherently costs carbon, while a forecast that
says nothing costs the ability to finish the work. The second is the sharper
statement of forecast dependence, because it is monotone and survives the
iso-completion filter that the carbon comparison does not.


% ---------------------------------------------------------------------
% BLOCK 3 -- auditor grid + calibrated trust rule.
% Paste into Appendix H (app:auditor), replacing nothing; add after the
% existing Table 5 discussion.
% Data: local_eval_rt/auditor_grid.txt, auditor_calibrated.txt
% (Vanilla seed 3, three episodes per cell, argmax, resid monitor).
% ---------------------------------------------------------------------
\begin{table}[t]
\centering
\caption{Auditor across the corruption family, one trained policy, three
episodes per cell. $\chi$ is the rolling forecast--supply correlation averaged
over the run. The class-default gate fires below $\chi{=}0.2$; the calibrated
gate fires below $\tfrac{1}{2}\chi_{\text{clean}}$ with $\chi_{\text{clean}}$
measured once on clean forecasts ($0.709$ here), giving $0.354$.
$\dagger$: carbon deflated by dropped work.}
\label{tab:auditor-grid}
\begin{tabular}{llrrrr}
\toprule
Forecast & Auditor & $\chi$ & Fired & Carbon & Completion (\%) \\
\midrule
clean            & off              & \phantom{-}0.708 & ---    & 0.185 & \phantom{0}99.9 \\
inverted         & off              & -0.708           & ---    & 0.126$^\dagger$ & \phantom{0}79.1 \\
inverted         & gate             & -0.708           & 99.4\% & 0.203 & \phantom{0}96.4 \\
inverted         & repair           & -0.708           & 99.4\% & \textbf{0.178} & \textbf{\phantom{0}99.9} \\
neutralised      & off              & \phantom{-}0.000 & ---    & 0.131$^\dagger$ & \phantom{0}95.9 \\
neutralised      & gate             & \phantom{-}0.000 & 99.4\% & 0.182 & \phantom{0}98.8 \\
neutralised      & repair           & \phantom{-}0.000 & \phantom{0}0.0\% & 0.125$^\dagger$ & \phantom{0}95.6 \\
reassigned       & off              & \phantom{-}0.230 & ---    & 0.226 & 100.0 \\
reassigned       & gate (default)   & \phantom{-}0.230 & 14.9\% & 0.219 & 100.0 \\
reassigned       & gate (calibrated)& \phantom{-}0.230 & 99.4\% & \textbf{0.214} & 100.0 \\
clean            & gate (calibrated)& \phantom{-}0.708 & \phantom{0}0.5\% & 0.188 & 100.0 \\
\bottomrule
\end{tabular}
\end{table}

Three readings follow. First, the chain from detection to restoration closes
only on the inversion: gating recovers most of the lost completion, and
repairing the flagged features returns the policy to its clean operating point
on both axes ($0.178$ against $0.185$ clean, 99.9\% completion). Second, repair
is correctly silent under neutralisation --- an uninformative constant carries
nothing to invert, so gating is the only available response, and the auditor
distinguishes the two cases without being told which corruption it faces.
Third, and least comfortably, the class-default threshold is calibrated for
sign inversion and nearly misses the reassignment that Table~\ref{tab:main}
leads with: a permutation preserves each site's curve and only mis-addresses
it, so the per-site correlation is diluted to $0.230$ rather than driven
negative, and the gate fires on 14.9\% of decisions. Expressing the same rule
relative to a clean calibration rather than as an absolute constant covers all
three corruptions --- the gate then fires on 99.4\% of reassigned decisions and
lowers their carbon to $0.214$ --- while leaving clean operation alone, where it
fires on 0.5\% of decisions at a $1.8\%$ carbon cost. The absolute form remains
the default; the calibrated form is what a deployment with a measured
$\chi_{\text{clean}}$ should use.


% ---------------------------------------------------------------------
% BLOCK 4 -- ablation upgrade.  Replaces the single-seed sentence in the
% Section 4 ablation paragraph.  Data: local_eval_rt/abl_*_ck*_*.log,
% checkpoint chosen by the Table 1 rule (lowest clean carbon among cells
% completing >= 99.5% clean).
% ---------------------------------------------------------------------
An ablation isolates the component behind the containment, now on two seeds per
variant. Fixing the gate open ($c_t\equiv1$) leaves clean carbon unchanged
($0.185$ and $0.182$ against $0.192$) but erases the corruption advantage on
both seeds, shuffle carbon rising to $0.282$ and $0.275$ --- past Vanilla's
$0.269$ in both cases. Removing the reweighting instead leaves shuffle carbon at
$0.201$ and $0.224$, a spread that straddles the full method and stays inside
the campaign's measured noise floor, so that component is not resolved at this
sample size. The gate, not the decomposition alone, is what turns credit
attribution into robustness, and it is the one component whose removal reverses
the sign on every seed measured.
