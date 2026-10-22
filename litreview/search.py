#!/usr/bin/env python3
"""Systematic literature search for the Probe-Trust paper (OpenAlex + Scopus). Keys are read from ~/.config/lit-watch.env and are never printed.
Queries are fixed in advance (QUERIES), results go to litreview/hits_<id>.jsonl, the summary to litreview/search_log.md. Usage: python3 search.py"""
import json, os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
H = Path(__file__).parent
def key(name):
    for l in open(os.path.expanduser("~/.config/lit-watch.env")):
        if l.startswith(name + "="): return l.split("=", 1)[1].strip().strip('"\'')
OA, EL = key("OPENALEX_API_KEY"), key("ELSEVIER_API_KEY")
Q = {  # id: (description, OpenAlex title_and_abstract.search, Scopus TITLE-ABS-KEY)
 "A1": ("FL poisoning + synthetic/probe inputs on the server", '"federated learning" AND (poisoning OR byzantine) AND ("synthetic data" OR "synthetic samples" OR probe OR probing OR "data-free" OR "without validation data" OR "without clean data")',
        'TITLE-ABS-KEY("federated learning" AND (poisoning OR byzantine) AND ("synthetic data" OR "synthetic samples" OR probe OR probing OR "data-free" OR "without validation data"))'),
 "A2": ("FL poisoning: trust/reputation in transport", '"federated learning" AND (poisoning OR byzantine OR malicious) AND (trust OR reputation) AND (traffic OR vehicular OR "intelligent transportation")',
        'TITLE-ABS-KEY("federated learning" AND (poisoning OR byzantine OR malicious) AND (trust OR reputation) AND (traffic OR vehicular OR "intelligent transportation"))'),
 "A3": ("federated traffic forecasting + attacks", '(federated) AND ("traffic forecasting" OR "traffic prediction" OR "traffic flow prediction") AND (poisoning OR attack OR byzantine OR malicious OR robust)',
        'TITLE-ABS-KEY(federated AND ("traffic forecasting" OR "traffic prediction" OR "traffic flow prediction") AND (poisoning OR attack OR byzantine OR malicious))'),
 "A4": ("physical constraints for detecting attacks on traffic sensors", '(physics OR "physical constraint" OR "physics-informed") AND (traffic) AND (attack OR spoofing OR adversarial) AND (detection OR defense OR defence)',
        'TITLE-ABS-KEY((physics OR "physical constraint" OR "physics-informed") AND traffic AND (attack OR spoofing OR adversarial) AND (detection OR defense))'),
 "A5": ("LLM agents: Byzantine/trust/consensus", '("LLM agents" OR "multi-agent LLM" OR "large language model agents") AND (byzantine OR malicious OR compromised) AND (trust OR consensus OR robust)',
        'TITLE-ABS-KEY(("LLM agents" OR "multi-agent LLM" OR "large language model agents") AND (byzantine OR malicious OR compromised) AND (trust OR consensus))'),
 "A6": ("agents: canary/honeypot/verification tasks", '(agent OR agents) AND (LLM OR "large language model") AND (canary OR honeypot OR "probe tasks" OR "behavioral attestation" OR "verification probes")',
        'TITLE-ABS-KEY((agent OR agents) AND (LLM OR "large language model") AND (canary OR honeypot OR "behavioral attestation" OR "probe tasks"))'),
 "A7": ("FLTrust and its dependence on root data", '(FLTrust OR "trust bootstrapping" OR "root dataset") AND ("federated learning") AND (poisoning OR byzantine)',
        'TITLE-ABS-KEY((FLTrust OR "trust bootstrapping" OR "root dataset") AND "federated learning" AND (poisoning OR byzantine))'),
 "A8": ("agentic AI + federated learning + security", '("agentic AI" OR "agentic") AND ("federated learning") AND (security OR trust OR poisoning OR robust)',
        'TITLE-ABS-KEY(("agentic AI" OR agentic) AND "federated learning" AND (security OR trust OR poisoning))'),
}
def abstract(inv):
    if not inv: return ""
    w = [(p, k) for k, ps in inv.items() for p in ps]; return " ".join(k for _, k in sorted(w))
def oa(qid, search, pages=2):
    out, total = [], None
    for pg in range(1, pages + 1):
        p = {"filter": f"title_and_abstract.search:{search},from_publication_date:2017-01-01", "per_page": 50, "page": pg, "api_key": OA,
             "select": "id,doi,title,publication_year,cited_by_count,primary_location,authorships,abstract_inverted_index,type,open_access"}
        try:
            with urllib.request.urlopen(urllib.request.Request("https://api.openalex.org/works?" + urllib.parse.urlencode(p), headers={"User-Agent": "probe-trust-litreview"}), timeout=60) as r: d = json.load(r)
        except Exception as e: print(qid, "OpenAlex error", str(e)[:80]); break
        total = d["meta"]["count"]
        for w in d["results"]:
            out.append(dict(src="openalex", id=w["id"], doi=w.get("doi"), title=w["title"], year=w["publication_year"], cited=w["cited_by_count"], venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                            authors=[a["author"]["display_name"] for a in w["authorships"][:6]], abstract=abstract(w.get("abstract_inverted_index")), oa=(w.get("open_access") or {}).get("oa_url")))
    return out, total
def scopus(qid, query, pages=2):
    out, total = [], None
    for pg in range(pages):
        url = "https://api.elsevier.com/content/search/scopus?" + urllib.parse.urlencode({"query": query + " AND PUBYEAR > 2016", "count": 25, "start": pg * 25, "sort": "relevancy"})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"X-ELS-APIKey": EL, "Accept": "application/json"}), timeout=60) as r: d = json.load(r)["search-results"]
        except Exception as e: print(qid, "Scopus error", str(e)[:80]); break
        total = int(d.get("opensearch:totalResults", 0))
        for e in d.get("entry", []):
            if "error" in e: continue
            out.append(dict(src="scopus", doi=e.get("prism:doi"), title=e.get("dc:title"), year=(e.get("prism:coverDate") or "")[:4], cited=int(e.get("citedby-count") or 0), venue=e.get("prism:publicationName"), authors=[e.get("dc:creator")], abstract="", oa=None))
    return out, total
log = ["# Literature search log (OpenAlex + Scopus), 2026-10-20", "", "| id | description | OpenAlex found | Scopus found | exported (OA/Scopus) |", "|---|---|---|---|---|"]
for qid, (desc, s_oa, s_sc) in Q.items():
    a, ta = oa(qid, s_oa); b, tb = scopus(qid, s_sc)
    seen, merged = set(), []
    for r in a + b:
        k = (r["doi"] or r["title"] or "").lower().replace("https://doi.org/", "")
        if k in seen: continue
        seen.add(k); merged.append(r)
    with open(H / f"hits_{qid}.jsonl", "w") as f:
        for r in merged: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.append(f"| {qid} | {desc} | {ta} | {tb} | {len(a)}/{len(b)} → {len(merged)} unique |"); print(qid, desc, "OA", ta, "Scopus", tb, "unique", len(merged)); time.sleep(1)
open(H / "search_log.md", "w").write("\n".join(log) + "\n\nQueries (exact strings) are in search.py. Period: 2017-2026. Screening: titles and abstracts read manually (see screening.md).\n")
