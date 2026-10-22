"""LaTeX backend (IEEEtran class, journal): the same text as the docx backend. References are \\cite by key, the bibliography is thebibliography in the order of first mention.
Build: BACKEND=tex python build_paper.py -> tex/probe_trust.tex; then /mnt/projects/tools/tectonic/tectonic tex/probe_trust.tex."""
import os, re, shutil
from paper_refs import REF

OUT = "tex/probe_trust.tex"
ORDER = []; BODY = []
FIGN = [0]; TABN = [0]; EQN = [0]; SECN = [0]; SUBN = [0]
ROM = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII"]
CITE_L, CITE_R = "\u27e6", "\u27e7"

def C(*keys):
    for k in keys:
        if k not in ORDER: ORDER.append(k)
    return CITE_L + ",".join(keys) + CITE_R

GREEK = {"σ": r"\sigma", "δ": r"\delta", "τ": r"\tau", "θ": r"\theta", "β": r"\beta", "α": r"\alpha", "Σ": r"\sum", "ŷ": r"\hat{y}", "∈": r"\in", "×": r"\times", "−": "-", "±": r"\pm", "≤": r"\leq", "≥": r"\geq", "≈": r"\approx", "→": r"\rightarrow", "↔": r"\leftrightarrow"}
PLAIN = {"“": "``", "”": "''", "’": "'", "‘": "`", "–": "--", "—": "---", "…": r"\ldots{}", "\u00a0": "~", "✓": r"\checkmark{}"}
MANUAL = [  # exact fragments where indices/powers must be typed as a formula
    ("max(RMSE_event/RMSE_event^clean, RMSE_all/RMSE_all^clean) − 1", r"$\max(\mathrm{RMSE}_{\mathrm{event}}/\mathrm{RMSE}^{\mathrm{clean}}_{\mathrm{event}},\ \mathrm{RMSE}_{\mathrm{all}}/\mathrm{RMSE}^{\mathrm{clean}}_{\mathrm{all}})-1$"),
    ("θ = Σ_i w_i θ_i / Σ_i w_i", r"$\theta=\sum_i w_i\theta_i\big/\sum_i w_i$"),
    ("f(θ_i)", r"$f(\theta_i)$"),
    ("β ∈ {0.2, 0.4, 0.6}", r"$\beta\in\{0.2,0.4,0.6\}$"),
]

def esc(text):
    """Escaping: references [[...]] -> \\cite, formulas and Greek letters -> math, special characters -> LaTeX."""
    math = []
    def keep(s): math.append(s); return f"\u27ea{len(math)-1}\u27eb"
    for a, b in MANUAL: text = text.replace(a, keep(b))
    text = re.sub(r"(?<![A-Za-z])([A-Za-zσδτθβ])_([a-z]{1,2})\b", lambda m: keep("$" + GREEK.get(m.group(1), m.group(1)) + "_{" + m.group(2) + "}$"), text)
    cites = []
    def ck(m): cites.append(m.group(1)); return f"\u27e8{len(cites)-1}\u27e9"
    text = re.sub(CITE_L + r"([^" + CITE_R + "]*)" + CITE_R, ck, text)
    out = []
    for ch in text:
        if ch in GREEK: out.append(r"\ensuremath{" + GREEK[ch] + "}")
        elif ch in PLAIN: out.append(PLAIN[ch])
        elif ch == "%": out.append(r"\%")
        elif ch == "&": out.append(r"\&")
        elif ch == "#": out.append(r"\#")
        elif ch == "_": out.append(r"\_")
        elif ch == "{": out.append(r"\{")
        elif ch == "}": out.append(r"\}")
        elif ch == "~": out.append(r"\textasciitilde{}")
        elif ch == "^": out.append(r"\textasciicircum{}")
        else: out.append(ch)
    t = "".join(out)
    t = re.sub("\u27e8(\\d+)\u27e9", lambda m: "\\cite{" + cites[int(m.group(1))] + "}", t)
    t = re.sub("\u27ea(\\d+)\u27eb", lambda m: math[int(m.group(1))], t)
    bad = [c for c in t if ord(c) > 127]
    assert not bad, f"unmapped symbols {set(bad)} in: {text[:80]}"
    return t

def P(text="", bold=False, italic=False, size=None, align="j", indent=True, after=None, caps=False, before=None):
    t = esc(text)
    if bold: t = r"\textbf{" + t + "}"
    if italic: t = r"\textit{" + t + "}"
    BODY.append(t + "\n\n")
def H1(t, num=True):
    SECN[0] += num; SUBN[0] = 0
    BODY.append(("\\section{" if num else "\\section*{") + esc(t) + "}\n")
def H2(t): SUBN[0] += 1; BODY.append("\\subsection{" + esc(t) + "}\n")
def mixed(parts, indent=True, size=None):
    if parts[0][0].startswith("Abstract"): BODY.append("\\begin{abstract}\n" + esc(parts[1][0]) + "\n\\end{abstract}\n"); return
    if parts[0][0].startswith("Index Terms"): BODY.append("\\begin{IEEEkeywords}\n" + esc(parts[1][0].rstrip(".")) + "\n\\end{IEEEkeywords}\n"); return
    s = ""
    for t, b, i in parts:
        e = esc(t); e = r"\textbf{" + e + "}" if b else e; e = r"\textit{" + e + "}" if i else e; s += e
    BODY.append(s + "\n\n")
def FIG(path, cap, width=17.0):
    FIGN[0] += 1; name = os.path.basename(path); os.makedirs("tex/figs", exist_ok=True); shutil.copy(path, "tex/figs/" + name)
    BODY.append("\\begin{figure*}[!t]\n\\centering\n\\includegraphics[width=" + f"{min(width, 17.4):.1f}" + "cm]{figs/" + name + "}\n\\caption{" + esc(cap) + "}\n\\end{figure*}\n"); return FIGN[0]
def TAB(cap, rows, widths=None, fs=7.5, bold_rows=()):
    TABN[0] += 1; n = len(rows[0]); widths = list(widths or [17.0 / n] * n); k = min(1.0, 17.0 / sum(widths)); widths = [w * k for w in widths]
    spec = "|".join(">{\\raggedright\\arraybackslash}p{" + f"{w - 0.25:.2f}" + "cm}" for w in widths)
    s = "\\begin{table*}[!t]\n\\caption{" + esc(cap) + "}\n\\centering\n\\footnotesize\n\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{|" + spec + "|}\n\\hline\n"
    for i, row in enumerate(rows):
        cells = [esc(str(c)) for c in row]
        if i == 0 or i in bold_rows: cells = [r"\textbf{" + c + "}" for c in cells]
        s += " & ".join(cells) + " \\\\\n\\hline\n"
    BODY.append(s + "\\end{tabular}\n\\end{table*}\n"); return TABN[0]
EQS = [r"v_i=\operatorname*{mean}_k\max\!\bigl(0,\hat y_{i,k}-s_k-\delta\bigr)+\operatorname*{mean}_k\max\!\bigl(0,u_k-\delta-\hat y_{i,k}\bigr)",
       r"w_i=\exp(-v_i/\tau),\qquad \theta=\sum_i w_i\theta_i\Big/\sum_i w_i"]
def EQ(parts):
    BODY.append("\\begin{equation}\n" + EQS[EQN[0]] + "\n\\end{equation}\n"); EQN[0] += 1; return EQN[0]
def FRONT(title, authors, affil):
    BODY.append("\\title{" + esc(title) + "}\n\\author{\\IEEEauthorblockN{" + esc(authors) + "}\n\\IEEEauthorblockA{" + esc(affil) + "}}\n\\maketitle\n")
def FINISH():
    unused = [k for k in REF if k not in ORDER]; assert not unused, f"unused sources: {unused}"
    os.makedirs("tex", exist_ok=True)
    pre = ("\\documentclass[journal]{IEEEtran}\n\\usepackage{graphicx}\n\\usepackage{array}\n\\usepackage{amsmath,amssymb}\n\\usepackage{url}\n\\usepackage{cite}\n"
           "\\hyphenation{Probe-Trust fed-er-at-ed}\n\\begin{document}\n")
    bib = "\\begin{thebibliography}{99}\n" + "".join("\\bibitem{" + k + "} " + esc(REF[k]) + "\n" for k in ORDER) + "\\end{thebibliography}\n"
    open(OUT, "w").write(pre + "".join(BODY) + "\n" + bib + "\\end{document}\n"); print("done", OUT, "references:", len(ORDER), "figures:", FIGN[0], "tables:", TABN[0])
