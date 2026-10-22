"""Bibliography check against OpenAlex: for each reference the found title, authors, year, venue and DOI. Result: verify_refs.md."""
import json, os, urllib.parse, urllib.request, time
def key():
    for l in open(os.path.expanduser("~/.config/lit-watch.env")):
        if l.startswith("OPENALEX_API_KEY="): return l.split("=",1)[1].strip().strip('"\'')
K = key()
REFS = [
 ("mcmahan17", "Communication-Efficient Learning of Deep Networks from Decentralized Data", 2017),
 ("krum17", "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", 2017),
 ("yin18", "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", 2018),
 ("fltrust21", "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping", 2021),
 ("rohfl23", "Toward Robust Hierarchical Federated Learning in Internet of Vehicles", 2023),
 ("securegraphfl25", "SecureGraphFL: A Privacy-Preserving and Attack-Resilient Federated Learning Framework for Traffic Prediction", 2025),
 ("wireless24", "Poisoning Attacks on Federated Learning-based Wireless Traffic Prediction", 2024),
 ("flare25", "FLARE: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning", 2025),
 ("dcrnn18", "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", 2018),
 ("stgcn18", "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting", 2018),
 ("cpwbft25", "Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance", 2025),
 ("sac26", "Robust Multi-Agent LLMs under Byzantine Faults", 2026),
 ("atrust25", "To trust or not to trust: Attention-based Trust Management for LLM Multi-Agent Systems", 2025),
 ("springdrift26", "Springdrift: An Auditable Persistent Runtime for LLM Agents with Case-Based Memory, Normative Safety, and Ambient Self-Perception", 2026),
 ("trustprop26", "Trust propagation and structural containment in Multi-agent LLM pipelines", 2026),
 ("rsmooth25", "Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing", 2025),
 ("survey21", "A survey on federated learning", 2021),
 ("bulyan18", "The Hidden Vulnerability of Distributed Learning in Byzantium", 2018),
 ("normbound19", "Can You Really Backdoor Federated Learning?", 2019),
 ("foolsgold20", "The Limitations of Federated Learning in Sybil Settings", 2020),
 ("alie19", "A Little Is Enough: Circumventing Defenses For Distributed Learning", 2019),
 ("qwen3", "Qwen3 Technical Report", 2025),
 ("zafar25", "Byzantine-Robust Federated Learning Using Generative Adversarial Networks", 2025),
 ("fedfg26", "FedFG: Privacy-Preserving and Robust Federated Learning via Flow-Matching Generation", 2026),
 ("vettraffic26", "Revisiting the Adversarial Robustness of Graph-Based Traffic Forecasting", 2026),
 ("fedmid24", "FedMID: A Data-Free Method for Using Intermediate Outputs as a Defense Mechanism Against Poisoning Attacks in Federated Learning", 2024),
 ("specshield26", "Frequency-Domain Signatures for Proactive Defense Against Model Poisoning Attacks in Federated Learning", 2026),
 ("uav26", "Receiver-Domain Behavioral Probing for Backdoor-Resilient Federated GPS Spoofing Detection in UAV Networks", 2026),
 ("danilenka24", "Tackling Non-IID Data and Data Poisoning in Federated Learning using Adversarial Synthetic Data", 2024),
 ("mudguard24", "MUDGUARD: Taming Malicious Majorities in Federated Learning using Privacy-preserving Byzantine-robust Clustering", 2024),
 ("flest23", "Better Safe Than Sorry: Constructing Byzantine-Robust Federated Learning with Synthesized Trust", 2023),
 ("traftrust26", "TrafTrust-Fed: Traffic-Trust Federated Learning with Fuzzy Inference", 2026),
 ("zenopp20", "Zeno++: Robust Fully Asynchronous SGD", 2020),
 ("snow08", "Cheap and Fast - But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks", 2008),
 ("p4p26", "P4P: A probe-guided anti-poisoning defense for federated learning-based intrusion detection in IoT networks under non-IID data", 2026),
 ("medsecfed22", "MediSecFed: Private and Secure Medical Image Classification in the Presence of Malicious Clients", 2022),
]
out = ["# Bibliography check against OpenAlex (2026-10-20)", "", "| key | requested | found (year, venue) | authors | DOI | match |", "|---|---|---|---|---|---|"]
res = {}
for k, t, y in REFS:
    p = {"search": t, "per_page": 3, "api_key": K, "select": "title,publication_year,primary_location,authorships,doi,cited_by_count"}
    try:
        with urllib.request.urlopen(urllib.request.Request("https://api.openalex.org/works?" + urllib.parse.urlencode(p), headers={"User-Agent": "probe-trust-verify"}), timeout=60) as r: d = json.load(r)["results"]
    except Exception as e: out.append(f"| {k} | {t[:50]} | ERROR {str(e)[:40]} | | | |"); continue
    best = None
    for w in d:
        a = "".join(c for c in (w["title"] or "").lower() if c.isalnum()); b = "".join(c for c in t.lower() if c.isalnum())
        if a[:40] == b[:40]: best = w; break
    w = best or (d[0] if d else None)
    if not w: out.append(f"| {k} | {t[:50]} | NOT FOUND | | | ✗ |"); continue
    ok = "✓" if best else "?"
    ven = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    au = ", ".join(a["author"]["display_name"] for a in w["authorships"][:5]) + (" …" if len(w["authorships"]) > 5 else "")
    res[k] = dict(title=w["title"], year=w["publication_year"], venue=ven, authors=au, doi=w["doi"], match=ok)
    out.append(f"| {k} | {t[:45]}… ({y}) | {w['title'][:55]} ({w['publication_year']}, {ven}) | {au} | {w['doi']} | {ok} |"); time.sleep(0.3)
open("verify_refs.md", "w").write("\n".join(out)); json.dump(res, open("verify_refs.json", "w"), ensure_ascii=False, indent=1)
for l in out[4:]: print(l[:320])
