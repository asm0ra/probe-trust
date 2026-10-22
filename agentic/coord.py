"""Baseline method "LLM coordinator": the same model receives the 10 forecasts and issues a final one. Shares 0.4 and 0.6, 3 draws of the ensemble per window. Cache out/coord_<attack>.jsonl."""
import json, os, re, sys, time, urllib.request, concurrent.futures as cf
import numpy as np
from lib import *
URL = os.environ.get("LLM_URL", "http://127.0.0.1:8099/v1/chat/completions")
def call(hist, fs):
    lines = "\n".join(f"Agent {i+1}: {list(np.round(f, 1))}" for i, f in enumerate(fs))
    user = f"Ten agents proposed 15-minute speed forecasts (mph, at +5, +10, +15 minutes) for the same highway sensor. Its last 60 minutes of readings (oldest first): {hist}.\n{lines}\nSome agents may be compromised or wrong. Decide the final forecast. Answer with JSON only: {{\"forecast\": [a, b, c]}}."
    sy = "You are the coordinator agent of a traffic forecasting system."
    msgs = [dict(role="user", content=sy + "\n\n" + user)] if os.environ.get("NO_SYSTEM") else [dict(role="system", content=sy), dict(role="user", content=user)]
    body = dict(messages=msgs, temperature=0.0, max_tokens=40, seed=1, cache_prompt=True)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]; m = re.search(r"\[([^\]]+)\]", r)
    try: f = [float(x) for x in m.group(1).split(",")][:3]; return f if len(f) == 3 else None
    except Exception: return None
if __name__ == "__main__":
    attack = sys.argv[1]; H, A = load("honest"), load(attack); items = [w["id"] for w in WIN]; FH, FA = forecasts(H, items), forecasts(A, items)
    jobs = [(share, j, r) for share in (0.4, 0.6) for j in range(len(WIN)) for r in range(3)]; out = f"{OUT_DIR}/coord_{attack}.jsonl"
    done = {(json.loads(l)["share"], json.loads(l)["j"], json.loads(l)["r"]) for l in open(out)} if os.path.exists(out) else set(); jobs = [x for x in jobs if x not in done]; fh = open(out, "a"); t0 = time.time()
    def work(job):
        share, j, r = job; h, a = comp(attack, share, WIN[j]["id"], r); fs = np.vstack([FH[h, j], FA[a, j]]); fs = fs[~np.isnan(fs).any(1)]
        rng = np.random.default_rng(j * 31 + r); fs = fs[rng.permutation(len(fs))]
        return dict(share=share, j=j, r=r, forecast=call(WIN[j]["hist"], fs))
    with cf.ThreadPoolExecutor(4) as ex:
        for n, res in enumerate(ex.map(work, jobs), 1):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if n % 100 == 0: print(attack, n, len(jobs), f"{time.time()-t0:.0f} s", flush=True)
    print("done", attack, len(jobs))
