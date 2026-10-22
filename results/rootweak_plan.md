# Plan "weak FLTrust root" (written BEFORE the computations, 2026-10-20)

Question: is there a scenario in which Probe2 / Probe2+FLTrust is honestly better than FLTrust (a contribution is needed for Q1)? Previous result: with a clean root of 10 nodes FLTrust has <= 3 % harm, Probe2+FLTrust is about FLTrust.

Hypothesis: FLTrust depends on the server's root update D0; Probe2 (no root) does not. So with a weak root FLTrust gets worse and Probe2 does not (but Probe2 is weak on flip), and Probe2+FLTrust should be no worse than the better of the two.

Root scenarios (labels R*):
- R_small2, R_small5: a root of 2 / 5 nodes (nested subsets of the same 10 random nodes, seed 4242).
- R_bias10: a "biased" root: 10 neighbouring nodes of the graph (ref.cluster, seed 4242), i.e. one district.
- R_poison: a root of 10 nodes, 3 of 10 have poisoned labels in the style of the attack (hide/exagg/bias, like the attackers; for flip/backdoor a general shift of +25 mph).
- N_iid0: non-IID clients (each client trains on 30 % of the days; the root as usual, 10 nodes).
- R_ctrl: control = the previous behaviour (root 10, IID), to check reproduction.
Methods: fltrust, probe2, probe2_fl, fedavg (reference). Attacks: hide, exagg, bias, flip (+backdoor if time remains). Shares 0.4 and 0.6. Seeds 0-4. Normalised thresholds delta=0.5 sigma, tau=0.1 sigma (norm_choice.json, not to be changed).
Order: METR-LA -> if there is a signal -> PEMS-BAY and PeMS-D7(M) with the same settings.

Success criterion (in advance): in scenario S and attack A, Probe2 or Probe2+FLTrust is better than FLTrust by the median harm, and the difference exceeds the confidence interval over seeds (paired difference by seed, bootstrap 95 % CI does not include 0), in >= 2 of 3 networks; at the same time on clean data no method breaks (|loss| <= 1 pp relative to FedAvg) in the same scenario. Harm = max(RMSE_event, RMSE_all)/clean-FedAvg(network) - 1, as in the previous analysis.
If the criterion cannot be reached, we record a negative result. All configurations that were run enter the report, including the failures. Method parameters are not changed after the computations started.
