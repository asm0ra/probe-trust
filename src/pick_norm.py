"""Choice of the normalised (delta, tau) from results/norm_tune.jsonl (METR-LA only, seeds 100-101): the minimum of the worst harm over two attacks (hide, exagg; share 0.4) with an overhead on clean data <= 3%."""
import json, pandas as pd
d = pd.read_json("../results/norm_tune.jsonl", lines=True)
base = d[(d.method == "fedavg") & (d.condition == "clean")].groupby("attack")[["rmse_event_honest", "rmse_all_honest"]].mean().mean()      # clean FedAvg
d["h"] = (d[["rmse_event_honest", "rmse_all_honest"]].div(base).max(axis=1) - 1) * 100
p = d[d.method == "probe2"].groupby(["delta", "tau", "attack", "condition"]).h.mean().unstack(["attack", "condition"])
p["worst"] = p[[("hide", "atk"), ("exagg", "atk")]].max(axis=1); p["clean"] = p[[("hide", "clean"), ("exagg", "clean")]].mean(axis=1)
p.columns = ["_".join(c).strip("_") for c in p.columns]; ok = p[p.clean <= 3.0]; ok = ok if len(ok) else p
cand = ok[ok.worst <= ok.worst.min() + 1.0]      # selection rule (fixed in advance): all variants within 1 pp of the best worst-case harm -> take the largest delta (a margin for natural variability of networks), then the largest tau
dk, tk = cand.sort_index(level=[0, 1]).index[-1]; best = ok.loc[(dk, tk)]
print(p.round(1).to_string()); print("candidates:", list(cand.index)); print("SELECTED delta=%.2f sigma, tau=%.2f sigma: worst harm %.1f %%, clean %.1f %%" % (dk, tk, best.worst, best.clean))
json.dump({"delta": dk, "tau": tk, "worst_harm_pct": float(best.worst), "clean_overhead_pct": float(best.clean), "rule": "within 1 pp of best worst-case harm -> largest delta, then largest tau; tuned on METR-LA seeds 100-101 only"}, open("../results/norm_choice.json", "w"))
