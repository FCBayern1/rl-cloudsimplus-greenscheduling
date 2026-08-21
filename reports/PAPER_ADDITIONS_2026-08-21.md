% =====================================================================
% Paper additions, 2026-08-21 (5080). Two blocks, both replacements.
% Every number is measured; provenance in the comment above each block.
% Nothing here is a new claim -- both blocks REPLACE existing text whose
% numbers came from a different training campaign or a thinner grid.
% =====================================================================


% =====================================================================
% BLOCK 1 -- ABLATION.  Replaces the single-seed, cross-campaign ablation
% sentence in Section 4 and adds a new appendix.
%
% WHY THIS IS A REPLACEMENT, NOT AN ADDITION:
% run_ablation.sh trained two single-key variants (ablG, ablW) off the
% knSV3b configuration but never evaluated knSV3b itself, so the paper
% compared ablG's 0.185 against Table 1's EU-CRD 0.192 -- a DIFFERENT
% training campaign (creg_eucrd_s2 / eucrd_v4). Config diff verified
% key-by-key 2026-08-21: ablG = knSV3b + blender.fixed_c=1.0, ablW =
% knSV3b + responsibility.reweight_advantages=false, nothing else.
% The base arm (logs/v3ht_knSb_s1,s2) was evaluated 2026-08-21 through
% the identical protocol: last 3 checkpoints x {clean,blend,shuffle},
% argmax, 1 episode, --seed = training seed. 18 cells.
% Data: local_eval_rt/local_rt_summary.txt, [abl ablBase_*] lines.
% Checkpoint rule = the paper's own: per seed, lowest clean carbon among
% checkpoints with clean completion >= 99.5%.
%   ablBase s1->ck8, s2->ck9 ; ablG s1->ck9, s2->ck10 ; ablW s1->ck9, s2->ck10
% =====================================================================

% ---- 1a. Section 4 replacement -------------------------------------
% FIND (the existing ablation paragraph, one sentence before the C_min one):
%
%   An ablation isolates the component behind the containment. Fixing the
%   gate open ($c_t\equiv1$) leaves clean carbon unchanged (0.185 against
%   0.192, single seed) but erases the corruption advantage, shuffle carbon
%   rising to 0.278, past even Vanilla. The gate, not the decomposition
%   alone, turns credit attribution into robustness.
%
% REPLACE WITH:

An ablation isolates the component behind the containment. Both variants are
single-key changes to one training configuration and are measured against that
configuration rather than against Table~\ref{tab:main}, so all three arms share
a campaign and a protocol (Appendix~\ref{app:ablation}). Fixing the gate open
($c_t\equiv1$) leaves clean carbon unchanged, 0.184 against the base 0.189, but
erases the corruption advantage: shuffle carbon rises to 0.279, past even
Vanilla, on both seeds. It is also the only arm whose shuffle completion leaves
the contract, so that figure is if anything deflated by dropped work. Removing
the advantage reweighting instead moves shuffle carbon by 6\%, inside the
checkpoint-to-checkpoint spread of this testbed, so at two seeds its separate
contribution is not resolvable. The gate, not the decomposition alone, turns
credit attribution into robustness.


% ---- 1b. New appendix ----------------------------------------------
% PASTE as a new appendix section, suggested position: after
% \section{Training Diagnostics} (app:mechanism) and before the auditor one.

\section{Component Ablation}
\label{app:ablation}

Table~\ref{tab:ablation} reports the two component ablations against the
configuration they are derived from. Each variant differs from the base in
exactly one configuration key, the base arm is evaluated through the same
protocol as the variants, and no cell is compared across training campaigns.

\begin{table}[ht]
\caption{Component ablation, carbon per completed work (deterministic decoding). Medians over two seeds; each seed contributes the checkpoint with the lowest clean carbon among those completing $\geq$99.5\% on clean forecasts, the rule used in Table~\ref{tab:main}. $^{\dagger}$: at least one seed below the completion contract, carbon deflated by dropped work.}
\label{tab:ablation}
\centering
\small
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
Arm & Clean & Blend & Shuffle \\
\midrule
Base (full method)                  & 0.189 & 0.186 & \textbf{0.227} \\
Gate open, $c_t\equiv1$             & 0.184 & 0.188 & 0.279$^{\dagger}$ \\
No advantage reweighting            & 0.174 & 0.185 & 0.213 \\
\bottomrule
\end{tabular*}
\end{table}

Two readings follow, one firm and one negative. Opening the gate costs nothing
on clean forecasts and 23\% on shuffled ones, and the effect reproduces on both
seeds separately (0.283 and 0.275 against 0.205 and 0.248). It is also the only
arm that drops work under shuffle, 99.46\% and 99.42\% against 100\% elsewhere,
so its carbon is measured on a slightly smaller workload than the arms it is
being compared with and the true gap is wider than the table shows.

Removing the advantage reweighting is the negative. Shuffle carbon moves from
0.227 to 0.213, a 6\% change in the direction that favours the ablated arm, on
two seeds at one episode per cell. The checkpoint-to-checkpoint spread of this
testbed is 10 to 13\%, so this difference is not resolvable and the table
records it as such. The claim the ablation supports is therefore specific: the
epistemic gate is load-bearing, and the reweighting is not shown to contribute
separately at this sample size.


% =====================================================================
% BLOCK 2 -- RUNTIME AUDITOR.  Replaces tab:auditor (4 rows, one
% corruption) and the "Three readings follow" paragraph in app:auditor.
%
% Data: local_eval_rt/auditor_grid.txt (Vanilla seed 3, 3 episodes,
% argmax, resid chi, 10 cells) and local_eval_rt/auditor_calibrated.txt
% (same arm, relative threshold 0.5 x chi_clean = 0.354).
%
% REPRODUCIBILITY: the inverted rows reproduce the independent 10-episode
% run currently in the paper to within 2.3% on carbon and 0.9 pp on
% completion (0.1262/79.09 vs 0.126/80.0 off; 0.2031/96.39 vs 0.205/96.9
% gate; 0.1778/99.91 vs 0.182/99.9 repair; 0.1846/99.92 vs 0.184/100.0
% clean). Stated in the caption so the thinner per-cell episode count is
% not a silent downgrade.
%
% NOISE PROBE (honest, and it bounds what may be claimed): two cells that
% are behaviourally identical -- repair mode on blend and on pshuffle
% never fires, so it degrades to `off' -- differ by 4.6% and 0.7% in
% carbon. The 3-episode carbon noise on this grid is therefore a few
% percent, which is why the pshuffle claims below are made on FIRE RATE
% and COMPLETION, not on carbon.
% =====================================================================

% ---- 2a. Table replacement -----------------------------------------
% FIND the existing \begin{table}[ht] ... \end{table} block carrying
% \label{tab:auditor}, and REPLACE the whole block with:

\begin{table}[ht]
\caption{Runtime auditor across three corruptions, one trained policy, three episodes per cell, deterministic decoding. Gate suppresses deferral once the cross-DC mean correlation $\chi$ falls below the threshold; repair additionally inverts a datacentre's forecast features under strong negative correlation. Fire is the fraction of decisions the gate acted on. The inverted rows reproduce an independent ten-episode run to within 2.3\% on carbon and 0.9\,pp on completion. $^{\dagger}$: carbon deflated by dropped work, not comparable.}
\label{tab:auditor}
\centering
\small
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
Forecast & Auditor & $\chi$ & Fire (\%) & Carbon & Completion (\%) \\
\midrule
clean (anchor) & off    & $+0.71$ & 0.0  & 0.185 & 99.9 \\
\addlinespace
inverted       & off    & $-0.71$ & 0.0  & 0.126$^{\dagger}$ & 79.1 \\
inverted       & gate   & $-0.71$ & 99.4 & 0.203 & 96.4 \\
inverted       & repair & $-0.71$ & 99.4 & \textbf{0.178} & \textbf{99.9} \\
\addlinespace
neutralised    & off    & $\phantom{+}0.00$ & 0.0  & 0.131$^{\dagger}$ & 95.9 \\
neutralised    & gate   & $\phantom{+}0.00$ & 99.4 & 0.182 & 98.8 \\
\addlinespace
reassigned     & off    & $+0.23$ & 0.0  & 0.226 & 100.0 \\
reassigned     & gate ($\theta{=}0.2$)   & $+0.23$ & 14.9 & 0.219 & 100.0 \\
reassigned     & gate (calibrated) & $+0.23$ & 99.4 & 0.214 & 100.0 \\
\bottomrule
\end{tabular*}
\end{table}

% ---- 2b. Text replacement ------------------------------------------
% FIND the paragraph beginning "Three readings follow." and running to
% "...which is the property Section~\ref{sec:auditor} claims."
% REPLACE WITH:

Four readings follow, and the last is a limitation. First, the forecast is
load-bearing at deployment: inverting it costs twenty percentage points of
completion on a policy that otherwise finishes the entire workload. Second, the
carbon of the unaudited inverted and neutralised cells is below the anchor only
because a fifth and a twentieth of the work is never done, the selection effect
Section~\ref{sec:experiments} guards against, so neither cell carries a carbon
claim. Third, on the inversion the auditor works and repair works better than
gating alone: suppressing deferral recovers most of the completion at a ten
percent carbon premium, whereas inverting the flagged features restores the
clean operating point on both axes at once, 0.178 against 0.185 clean and
99.9\% against 99.9\%. The chain from corruption to detection to restoration
closes on a single policy, which is the property Section~\ref{sec:auditor}
claims.

Fourth, the absolute threshold has a blind spot, and the correlation statistic
has one that calibration cannot reach. Under a reassigned forecast $\chi$
settles at $+0.23$, above the class default $\theta{=}0.2$, so the gate acts on
only 14.9\% of decisions and the corruption passes essentially unchallenged. The
statistic is not at fault, since $+0.23$ is far below the $+0.71$ a healthy
forecast produces; the constant is. Re-registering the threshold relative to the
clean operating point, at half of the $\chi$ the same policy shows on clean
forecasts, raises the fire rate to 99.4\% and holds completion at 100\%, at a
cost of firing on 0.5\% of clean decisions. The carbon differences among these
three reassigned cells are within the few-percent run-to-run spread of a
three-episode cell and no carbon claim is made on them; what the calibration
demonstrably fixes is detection, not saving. The residual limitation is
structural: blending a forecast towards a constant is an affine shrink, and
correlation is invariant to affine transformation, so $\chi$ stays near its
clean value under partial neutralisation and collapses only when the last of
the variance is gone. A trust statistic built on correlation therefore sees
coherent lies and complete silence, but not gradual loss of information, and
detecting that case needs a scale-sensitive statistic alongside it.
