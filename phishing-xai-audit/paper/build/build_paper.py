from docx.oxml import OxmlElement
"""Build the paper .docx from the IEEE template styles + the result files in ../../results.
Usage: python3 -I build_paper.py [out.docx]   (run make_figs.py first)"""
import copy, os, re, sys, json
import numpy as np, pandas as pd
import docx
from docx.oxml.ns import qn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nums import load, MODELS
from docx_helpers import clone, set_markup, picture_paragraph, make_table

W = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PAPER_DATA") or os.path.normpath(os.path.join(W, "..", "..", "results"))
FIGDIR = f"{W}/figs"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{W}/Phishing_XAI_IEEE_Paper.docx"
N = load()
# ---- RoBERTa-base: seed 42 (main run) + seeds 43/44 (separate Colab session, same recipe) -> three-seed mean/SD
_x = pd.read_csv(f"{DATA}/round2_r2res/roberta_reseeds.csv")      # seeds 43/44 retrained with eval-mode validation (round 2)
_s42 = N["all"][(N["all"].protocol == "dedup") & (N["all"].seed == 42) & (N["all"].model == "roberta-base")].iloc[0]
_neg, _pos = int(_s42.tn + _s42.fp), int(_s42.tp + _s42.fn)
def _mcc(r):
    fp = round(r.fpr * _neg); fn = round(r.fnr * _pos); tp = _pos - fn; tn = _neg - fp
    d = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))); return (tp*tn - fp*fn)/d
_x["mcc"] = _x.apply(_mcc, axis=1)
RO3 = {m: np.array([float(_s42[m])] + list(_x[m])) for m in ("acc", "f1", "fpr", "fnr", "mcc")}
RO3_SEEDS = [42] + list(_x.seed)
for m_, v_ in RO3.items():
    N["sd"].loc["roberta-base", m_] = v_.mean(); N["sd"].loc["roberta-base", m_ + ".1"] = v_.std(ddof=1)

F, G = N["F"], N["G"]
DI, RO = MODELS

# ------------------------------------------------------------------ numbers parsed from the audit log
AUD = re.sub(r"[ ]+", " ", open(f"{DATA}/audit_out.txt").read())
def rx(p, g=None):
    m = re.search(p, AUD); assert m, p
    return m.groups() if g is None else m.group(g)
n_groups, n_rows_groups = map(int, rx(r"groups with >1 copy: (\d+) \| rows in them: (\d+)"))
n_seen, acc_seen_lr = rx(r"seen in train n= *(\d+) acc=([\d.]+)"); n_seen = int(n_seen)
n_empty_seen = int(rx(r"seen, in the conflicting group\(s\) n= *(\d+)", 1))
n_other_seen, acc_other_seen = rx(r"seen, NOT in conflicting group n= *(\d+) acc=([\d.]+)"); n_other_seen = int(n_other_seen)
n_near, pct_near, _, acc_near = rx(r"(\d+) \(([\d.]+)%\)\n accuracy on those n= *(\d+) acc=([\d.]+)")
n_empty_all, n_empty_phish = map(int, rx(r"empty (\d+) (\d+)"))
na_safe, na_phish = [float(x) for x in rx(r"mean share of non-ASCII chars: \{0: ([\d.]+), 1: ([\d.]+)\}")]
hist = eval(rx(r"group-size histogram[^{]*(\{[^}]*\})", 1))
assert n_empty_seen == 104 and n_seen == 258 and n_other_seen == 154
a = N["audit"]; allr = N["all"]; allr1 = N["all1"]
def row(df, proto, seed, model): return df[(df.protocol == proto) & (df.seed == seed) & (df.model == model)].iloc[0]
def P(x, d=2): return f"{100 * x:.{d}f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
def pvs(p): return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"

# ------------------------------------------------------------------ references (numbered by first use)
REFS = {
 "uddin": "M. A. Uddin, M. Mahiuddin, and I. H. Sarker, “An explainable transformer-based model for phishing email detection: A large language model approach,” <i>Comput. Netw.</i>, vol. 277, Art. no. 112061, 2026, doi: 10.1016/j.comnet.2026.112061.",
 "jamal": "S. Jamal, H. Wimmer, and I. H. Sarker, “An improved transformer-based model for detecting phishing, spam and ham emails: A large language model approach,” <i>Secur. Privacy</i>, vol. 7, no. 5, 2024, doi: 10.1002/spy2.402.",
 "sajad": "Sajad U P, “A robust and explainable transformer-based framework for phishing email detection,” arXiv:2511.12085, 2025.",
 "lee": "K. Lee <i>et al.</i>, “Deduplicating training data makes language models better,” in <i>Proc. 60th Annu. Meeting Assoc. Comput. Linguistics</i>, 2022, pp. 8424–8445.",
 "kapoor": "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” <i>Patterns</i>, vol. 4, no. 9, Art. no. 100804, 2023.",
 "atanasova": "P. Atanasova, J. G. Simonsen, C. Lioma, and I. Augenstein, “A diagnostic study of explainability techniques for text classification,” in <i>Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)</i>, 2020, pp. 3256–3274.",
 "jacovi": "A. Jacovi and Y. Goldberg, “Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?” in <i>Proc. 58th Annu. Meeting Assoc. Comput. Linguistics</i>, 2020, pp. 4198–4205.",
 "krishna": "S. Krishna, T. Han, A. Gu, S. Wu, S. Jabbari, and H. Lakkaraju, “The disagreement problem in explainable machine learning: A practitioner’s perspective,” <i>Trans. Mach. Learn. Res.</i>, 2024.",
 "bastings": "J. Bastings, S. Ebert, P. Zablotskaia, A. Sandholm, and K. Filippova, “‘Will you find these shortcuts?’ A protocol for evaluating the faithfulness of input salience methods for text classification,” arXiv:2111.07367, 2021.",
 "deyoung": "J. DeYoung <i>et al.</i>, “ERASER: A benchmark to evaluate rationalized NLP models,” in <i>Proc. 58th Annu. Meeting Assoc. Comput. Linguistics</i>, 2020, pp. 4443–4458.",
 "boucher": "N. Boucher, I. Shumailov, R. Anderson, and N. Papernot, “Bad characters: Imperceptible NLP attacks,” in <i>Proc. IEEE Symp. Secur. Privacy (SP)</i>, 2022, pp. 1987–2004, doi: 10.1109/SP46214.2022.9833641.",
 "salloum": "S. Salloum, T. M. A. Gaber, S. Vadera, and K. Shaalan, “A systematic literature review on phishing email detection using natural language processing techniques,” <i>IEEE Access</i>, vol. 10, pp. 65703–65727, 2022, doi: 10.1109/ACCESS.2022.3183083.",
 "liu": "Y. Liu <i>et al.</i>, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.",
 "kaggle": "subhajournal, “Phishing email detection,” Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/subhajournal/phishingemails (accessed Oct. 2026).",
 "sanh": "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter,” arXiv:1910.01108, 2019.",
 "ribeiro": "M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’: Explaining the predictions of any classifier,” in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining</i>, 2016, pp. 1135–1144.",
 "lundberg": "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in <i>Adv. Neural Inf. Process. Syst.</i>, vol. 30, 2017, pp. 4765–4774.",
 "sundararajan": "M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in <i>Proc. 34th Int. Conf. Mach. Learn.</i>, 2017, pp. 3319–3328.",
 "kokhlikyan": "N. Kokhlikyan <i>et al.</i>, “Captum: A unified and generic model interpretability library for PyTorch,” arXiv:2009.07896, 2020.",
 "hooker": "S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, “A benchmark for interpretability methods in deep neural networks,” in <i>Adv. Neural Inf. Process. Syst.</i>, vol. 32, 2019.",
 "northcutt": "C. G. Northcutt, A. Athalye, and J. Mueller, “Pervasive label errors in test sets destabilize machine learning benchmarks,” in <i>Proc. 35th Conf. Neural Inf. Process. Syst. (NeurIPS), Datasets Benchmarks Track</i>, 2021.",
 "pedregosa": "F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine learning in Python,” <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.",
 "wolf": "T. Wolf <i>et al.</i>, “Transformers: State-of-the-art natural language processing,” in <i>Proc. Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations</i>, 2020, pp. 38–45.",
 "loshchilov": "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2019.",
 "pierse": "C. Pierse, “Transformers Interpret,” GitHub repository. [Online]. Available: https://github.com/cdpierse/transformers-interpret",
 "wilcoxon": "F. Wilcoxon, “Individual comparisons by ranking methods,” <i>Biometrics Bull.</i>, vol. 1, no. 6, pp. 80–83, 1945.",
 "holm": "S. Holm, “A simple sequentially rejective multiple test procedure,” <i>Scand. J. Statist.</i>, vol. 6, no. 2, pp. 65–70, 1979.",
 "metsis": "V. Metsis, I. Androutsopoulos, and G. Paliouras, “Spam filtering with naive Bayes—Which naive Bayes?” in <i>Proc. 3rd Conf. Email Anti-Spam (CEAS)</i>, 2006.",
 "enron_hf": "SetFit, “enron_spam,” Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/SetFit/enron_spam (accessed Oct. 2026).",
 "zefang": "zefang-liu, “phishing-email-dataset,” Hugging Face Datasets. [Online]. Available: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset (accessed Oct. 2026).",
 "mcnemar": "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” <i>Psychometrika</i>, vol. 12, no. 2, pp. 153–157, 1947.",
}
ORDER = []
def num(k):
    if k not in ORDER: ORDER.append(k)
    return ORDER.index(k) + 1
def c(*keys):
    ns = sorted({num(k) for k in keys})
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        if j - i >= 2: out.append(f"[{ns[i]}]–[{ns[j]}]")
        else: out.extend(f"[{n}]" for n in ns[i:j + 1])
        i = j + 1
    return ", ".join(out)

# ------------------------------------------------------------------ pull the numbers used in the text
def sd(model, met): return N["sd"].loc[model, met], N["sd"].loc[model, met + ".1"]
LR_, SV_, RF_ = "LR", "SVM", "RF"
d42 = {m: row(allr, "dedup", 42, m) for m in ["LR", "SVM", "RF", DI, RO]}
r42 = {m: row(allr, "raw", 42, m) for m in ["LR", "SVM", "RF", DI, RO]}
seenT = {m: row(allr, "raw_seen=True", 42, m) for m in MODELS}; seenF = {m: row(allr, "raw_seen=False", 42, m) for m in MODELS}
mc = N["mcn"].set_index("b"); pdf = N["paired"]
def pd_(m, b):
    r = pdf[(pdf.a == m) & (pdf.b == b)].iloc[0]; return r.diff_pts, r.ci_lo, r.ci_hi
rob = N["rob"]; rc = N["rc"]; ext = N["ext"]
def rb(m, atk): return rob[(rob.model == m) & (rob.attack == atk)].iloc[0]
def rcr(m, atk, ml=256): return rc[(rc.model == m) & (rc.attack == atk) & (rc.max_len == ml)].iloc[0]
tokm = rcr(DI, "none").mean_tokens; fr256 = rcr(DI, "none").frac_over_256
ft = N["faith_tab"].set_index(["model", "method"]); ft1 = N["faith_tab1"].set_index(["model", "method"])
alpha2 = {m: ft.loc[(m, "LITA-absmax(tuned)"), "alpha_tuned_absmax"] for m in MODELS}
alpha1 = {m: ft1.loc[(m, "LITA-absmax(tuned)"), "alpha_tuned_absmax"] for m in MODELS}
sh = N["shap"]
def shr(m, e): return sh[(sh.model == m) & (sh.evals == e)].iloc[0]
ag = N["ag"]
def comp(m, k): return F[(m, "comp", "mean")][k]
def suff(m, k): return F[(m, "suff", "mean")][k]
def pq(m, met, pair): return F[(m, met, "p")][pair][1]     # Holm-adjusted
ig = lambda m, v: G[(m, v)]
V_DEF, V_P300, V_L100, V_MASK, V_GXI = "IG_prob30 (paper)", "IG_prob300", "IG_logit100", "IG_logit100_maskbase", "GradxInput_logit"
xd = N["xd"].set_index(["dataset", "model"])
enr, zef = ext.loc["SetFit/enron_spam"], ext.loc["zefang-liu/phishing-email-dataset"]
max_acc_diff = max(abs(r.acc - allr1[(allr1.protocol == r.protocol) & (allr1.seed == r.seed) & (allr1.model == r.model)].acc.iloc[0])
                   for _, r in allr.iterrows())
floor_n = n_empty_seen - int(round(float(acc_seen_lr) * 0 + 64))     # 40 minority copies in the raw test split
floor_pct = 100 * 40 / 3727

def c1(m, k): return ft1.loc[(m, k), "comp_mean"]
def c2(m, k): return ft.loc[(m, k), "comp_mean"]
ratio_max = max(c1(DI, k) / c2(DI, k) for k in ("LIME", "SHAP", "IG", "LITA-absmax(a=.5)"))
ratio_lime = c1(DI, "LIME") / c2(DI, "LIME")
# ---- round-2 results (tuned/matched baselines, near-duplicates, "empty", Enron LR, LIME repeats, stratified sample)
R2D = f"{DATA}/round2_r2res"
_tuned = pd.read_csv(f"{R2D}/tuned_baselines.csv"); _match = pd.read_csv(f"{R2D}/matched_input.csv"); _mp = pd.read_csv(f"{R2D}/matched_paired.csv")
_ndl = pd.read_csv(f"{R2D}/neardup_leakage_dedup_seed42.csv"); _nds = pd.read_csv(f"{R2D}/neardup_group_stats.csv").set_index("thr"); _ndg = pd.read_csv(f"{R2D}/neardup_group_split.csv")
_emp = pd.read_csv(f"{R2D}/empty_sensitivity.csv"); _extc = pd.read_csv(f"{R2D}/external_classical.csv").set_index(["dataset", "model"])
_lr = pd.read_csv(f"{R2D}/lime_repeat_summary.csv"); _st = pd.read_csv(f"{R2D}/strat_faithfulness.csv")
def msd(df, model, col="acc", key="model"):
    v = 100 * df[df[key] == model][col].values; return v.mean(), v.std(ddof=1)
def fm(t): return f"{t[0]:.2f} ± {t[1]:.2f}"
def rng_(v): return f"{min(v):.2f}–{max(v):.2f}"
from scipy import stats as _stats
def boot_mean(x, n=2000, seed=0):
    r = np.random.RandomState(seed); x = np.asarray(x, float)
    return np.percentile([x[r.randint(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])
def TNR(el):
    for r in el.iter(qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr"); r.insert(0, rpr)
        if rpr.find(qn("w:rFonts")) is None:
            f = OxmlElement("w:rFonts")
            for a_ in ("ascii", "hAnsi", "cs", "eastAsia"): f.set(qn("w:" + a_), "Times New Roman")
            rpr.insert(0, f)
    return el
# ------------------------------------------------------------------ document skeleton from v2
doc = docx.Document(f"{W}/template/v2.docx")
body = doc.element.body
els = list(body.iterchildren())
PR = dict(H1=els[8], H2=els[18], BODY=els[9], BUL=els[12], FIGP=els[25], FIGCAP=els[26], TABHEAD=els[48],
          SPACER=els[50], REF=els[88], H5=els[87], EQ1=els[33], EQ2=els[35], EQ3=els[37], EQ4=els[41], EQ5=els[42])
PR = {k: copy.deepcopy(v) for k, v in PR.items()}
head = els[:8]; tail = els[-1]
for e in els[8:-1]: body.remove(e)

TITLE = "Beyond Accuracy: Auditing Duplicates, Baselines and Explanation Faithfulness in Transformer-Based Phishing Email Detection"
set_markup(head[0], TITLE)
from docx.oxml import parse_xml as _px
from docx.oxml.ns import nsdecls as _nsd
AUTHORS = [("Tellakula Muneiah", "muneiahtellakula@gmail.com"), ("Chandra Akhil", "akhilchandra018@gmail.com"),
           ("Nandulamatam Hemitha", "nandulamatamhemitha@gmail.com"), ("Chagaleti Tejeswar", "chagaletiteja75@gmail.com")]
def _ar(text, i=False, link=False):
    rpr = "<w:rPr>" + ("<w:i/>" if i else "") + ('<w:color w:val="0563C1"/>' if link else "") + '<w:sz w:val="18"/><w:szCs w:val="18"/>' + ('<w:u w:val="single"/>' if link else "") + "</w:rPr>"
    return f'<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r>'
_BR = '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:br/></w:r>'
def _author_cell(name, mail):
    rid = doc.part.relate_to("mailto:" + mail, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    runs = (_ar(name) + _BR + _ar("Department of Computer Science and Engineering [CS]", i=True) + _BR + _ar("RGMCET", i=True) + _BR + _ar("Andhra Pradesh, India") + _BR
            + f'<w:hyperlink r:id="{rid}" w:history="1">' + _ar(mail, link=True) + "</w:hyperlink>")
    return (f'<w:tc><w:tcPr><w:tcW w:w="5060" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="60" w:after="60"/><w:jc w:val="center"/></w:pPr>{runs}</w:p></w:tc>')
_rows = "".join("<w:tr>" + _author_cell(*AUTHORS[k]) + _author_cell(*AUTHORS[k + 1]) + "</w:tr>" for k in (0, 2))
_atbl = _px(f'<w:tbl {_nsd("w", "r")}><w:tblPr><w:tblW w:w="10120" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="5060"/><w:gridCol w:w="5060"/></w:tblGrid>' + _rows + "</w:tbl>")
head[2].addprevious(_atbl); body.remove(head[2])
def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o): run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj
_mp = _mp.copy(); _mp["adj"] = _holm(_mp.mcnemar_p.values)
_mpr = _mp[_mp.transformer == "roberta-base"]; MP_MAX = _mpr.diff_pts.max(); RO_HOLM = int((_mpr.adj < .05).sum())
_ND = [100 * _ndg[(_ndg.thr == t_) & (_ndg.model == m_)].acc.mean() - 100 * _tuned[_tuned.model == m_ + "_default"].acc.mean() for t_ in (0.95, 0.90) for m_ in ("LR", "SVM")]
ND_LO, ND_HI, ND_ABS = min(_ND), max(_ND), max(abs(x) for x in _ND)
_lrs = _lr.groupby("model").lime_comp.agg(["min", "max"]); LIME_RATIO = float((_lrs["max"] / _lrs["min"]).max())
ABSTRACT = (
    "High accuracy on public phishing-email benchmarks can hide data-quality problems and configuration-sensitive explanations. We audit the 18,631-email "
    "Kaggle Phishing Email corpus, compare TF-IDF baselines with DistilRoBERTa and RoBERTa-base, and test word-level "
    "explanations from LIME, SHAP, Integrated Gradients (IG) and a LITA-style LIME–IG hybrid by deletion. "
    f"One placeholder text, “empty” ({n_empty_all} copies, both labels), caps raw-split accuracy near {100 - floor_pct:.1f}%. Near-duplicate-aware group splits change TF-IDF accuracy by {ND_LO:+.2f} to {ND_HI:+.2f} points on different test sets (transformers: one seed). "
    f"On deduplicated data DistilRoBERTa and RoBERTa-base reach {P(N['sd'].loc[DI].acc,1)}% and {P(N['sd'].loc[RO].acc,1)}% accuracy (three-seed means) against {P(N['sd'].loc['SVM'].acc,1)}% for a TF-IDF SVM; "
    f"differences are not significant on full text. On matched 256-token inputs RoBERTa-base leads by up to {MP_MAX:.1f} percentage points, with {RO_HOLM} of its 12 comparisons remaining significant after Holm correction across the full family of 24 matched-input comparisons. "
    "Explanation rankings depended on configuration: IG on RoBERTa-base scored like random order with a pad baseline but matched LIME with a mask baseline, "
    "and SHAP’s score rose 2.5-fold with its evaluation budget. "
    f"Re-sampling LIME on fixed emails changed mean comprehensiveness at most {LIME_RATIO:.1f}-fold, whereas two pipeline runs differed up to {ratio_max:.1f}-fold, with sample and checkpoint effects not separated. "
    "The hybrid beat LIME on DistilRoBERTa but not RoBERTa-base (exploratory). "
    f"Homoglyphs raised DistilRoBERTa’s false-positive rate from {P(rb(DI,'none').fpr,1)}% to {P(rb(DI,'homoglyph').fpr,1)}%, not explained by truncation. "
    "A second public dataset mirrors the Kaggle corpus (99.8% overlap in our sample)."
)
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())
set_markup(head[6], "<i>Abstract</i>—" + ABSTRACT)
set_markup(head[7], "<i>Keywords</i>—phishing email detection, dataset auditing, explanation faithfulness, transformer evaluation, character-level robustness")

new = []
def H1(t): new.append(clone(PR["H1"], t))
def H2(t): new.append(clone(PR["H2"], t))
def Pp(t): new.append(clone(PR["BODY"], t))
def BUL(t): new.append(clone(PR["BUL"], t))
def EQ(k, markup=None): new.append(clone(PR[k]))
def FIG(path, w, cap):
    new.append(picture_paragraph(doc, PR["FIGP"], path, w)); new.append(clone(PR["FIGCAP"], cap))
def TAB(cap, tbl):
    new.append(clone(PR["TABHEAD"], cap)); new.append(tbl); new.append(clone(PR["SPACER"]))

# ================================================================== INTRODUCTION
H1("Introduction")
Pp(f"Phishing email remains a common route for credential theft and fraud. Transformer-based detectors have been applied to it {c('uddin','jamal')}, with accuracy above 98% reported on a public corpus {c('uddin')}, and post-hoc explanations such as LIME (local interpretable model-agnostic explanations) and SHAP (Shapley additive explanations) are increasingly attached to them so that an analyst can see why a message was flagged {c('uddin','sajad')}. Two assumptions behind such reports deserve explicit tests.")
Pp(f"The first is that the benchmark is clean. Corpora assembled from several sources can contain identical or near-identical emails on both sides of a random split, and duplicated data distorts evaluation in other domains {c('lee','kapoor')}. The second is that the explanation is trustworthy. Explainers differ in how faithfully they reflect the model {c('atanasova')}, a faithfulness claim needs an explicit test {c('jacovi')}, explainers disagree with one another {c('krishna')}, and the outcome of a faithfulness test can depend on how a method is configured {c('bastings')}. Uddin et al. {c('uddin')} proposed LITA, which averages min-max-normalised LIME and Transformers Interpret scores with <i>α</i> = 0.5 and illustrated it with word-level scores for two emails; no deletion-based or other quantitative faithfulness test was reported, and its term frequency–inverse document frequency (TF-IDF) baselines used default unigram features.")
Pp("This paper audits both assumptions on the same corpus and model family. It is an empirical audit, not a new detector or explanation method, and it asks four research questions:")
BUL("<b>RQ1 (benchmark integrity).</b> How do exact duplicates, near-duplicates and a conflicting-label placeholder affect the evaluation of phishing-email detectors, and are two public corpora independent of each other?")
BUL("<b>RQ2 (model comparison).</b> How do fine-tuned transformers compare with fixed and validation-tuned TF-IDF baselines on full-text and matched-input evaluations, and are the observed differences statistically significant?")
BUL(f"<b>RQ3 (explanation faithfulness).</b> How do deletion-based faithfulness rankings {c('deyoung')} of LIME, SHAP, Integrated Gradients (IG) and a LITA-style LIME–IG hybrid change with IG’s target, steps and baseline, SHAP’s evaluation budget and a repeat of the pipeline?")
BUL(f"<b>RQ4 (robustness).</b> How do homoglyph, typo and zero-width perturbations {c('boucher')} change false-positive and false-negative rates once truncation is controlled?")

# ================================================================== RELATED WORK
H1("Related Work")
Pp(f"Classical and deep-learning approaches to phishing email detection are surveyed in {c('salloum')}. Transformer encoders are now common: Jamal et al. {c('jamal')} fine-tuned BERT-family models for phishing and spam email, Uddin et al. {c('uddin')} fine-tuned RoBERTa {c('liu')} on the Kaggle Phishing Email corpus {c('kaggle')}, reporting 98.45% accuracy and comparing against TF-IDF models, and Sajad {c('sajad')} combined DistilBERT with adversarial training and random character noise and explained it with LIME, SHAP and IG. We study RoBERTa-base {c('liu')} and DistilRoBERTa, a RoBERTa model distilled with the procedure of {c('sanh')}.")
Pp(f"LIME {c('ribeiro')} fits a local surrogate on perturbed inputs, SHAP {c('lundberg')} assigns Shapley-value attributions, and IG {c('sundararajan')} integrates gradients along a path from a baseline input (implemented in Captum {c('kokhlikyan')}). Faithfulness is commonly tested by erasure: comprehensiveness measures the drop in the predicted probability when the highlighted words are removed, and sufficiency the drop when only they are kept {c('deyoung')}. Removal-based tests have their own confound, because the erased inputs are off-distribution {c('hooker')}. Atanasova et al. {c('atanasova')} compared gradient- and perturbation-based explainers on text classifiers, Jacovi and Goldberg {c('jacovi')} discuss what a faithfulness claim requires, Bastings et al. {c('bastings')} show that popular configurations of input-salience methods can perform poorly even on simple shortcuts and recommend checking each method for each task and model, and Krishna et al. {c('krishna')} document that practitioners meet conflicting explanations in practice.")
Pp(f"Lee et al. {c('lee')} show that duplicated training text affects language-model behaviour, Northcutt et al. {c('northcutt')} show that label errors in widely used benchmark test sets can change model rankings, and Kapoor and Narayanan {c('kapoor')} identify leakage as a cause of irreproducible machine-learning results. Boucher et al. {c('boucher')} show that imperceptible character-level changes, including homoglyphs and zero-width characters, can mislead text classifiers.")

# ================================================================== METHODOLOGY
H1("Methodology")
FIG(f"{FIGDIR}/fig1_pipeline.png", 3.2, "Study pipeline. The LIME–IG hybrid is evaluated against its component explainers on the same emails, and controls vary the explainers’ settings.")
H2("Data and Duplicate Audit")
Pp(f"Fig. 1 summarises the study pipeline. We use the public Kaggle Phishing Email corpus {c('kaggle')}, which {c('uddin')} also uses. It lists 18,650 emails; after removing rows with missing or empty text and stripping HTML tags and extra whitespace, {a['n_raw']:,} remain ({a['n_phish']:,} phishing, {a['n_safe']:,} safe). Two emails are exact duplicates if their text is identical after lower-casing and replacing every run of non-alphanumeric characters by one space. Near-duplicates are treated in two separate ways, and neither removes data from the primary protocol. In the overlap audit, a test email is a near-duplicate if its TF-IDF cosine similarity to some training email is at least 0.95, and in Section IV-A also at least 0.90; on the raw split we count only test emails with no exact copy in training, and on the dedup split every test email is checked. In the group-splitting control (Section III-B), emails linked at the same thresholds form connected components that are kept together in one partition. We evaluate two protocols: raw, a stratified random split of all emails (seed 42), and dedup, where only the first copy of each duplicate group is kept ({a['n_dedup']:,} emails, {a['n_dedup_phish']:,} phishing) and splits are repeated for seeds 42, 43 and 44. Splits are stratified 70/10/20 into training, validation and test; the raw test set has 3,727 emails and each dedup test set 3,499.")
H2("Detectors")
Pp(f"The TF-IDF baselines use word 1–2-grams (50,000 features, sublinear term frequency, minimum document frequency 2) with logistic regression (LR, <i>C</i> = 10), a linear SVM (<i>C</i> = 1) and a random forest (RF, 300 trees), all class-balanced and implemented with scikit-learn {c('pedregosa')} with hyper-parameters fixed in advance. DistilRoBERTa and RoBERTa-base are fine-tuned from the distilroberta-base and roberta-base checkpoints with the Transformers library {c('wolf')} using AdamW {c('loshchilov')} (learning rate 2×10<sup>−5</sup>, weight decay 0.01, batch size 32) for three epochs with 10% warm-up and linear decay, mixed precision, gradient-norm clipping at 1.0, class-weighted cross-entropy and inputs truncated to the first 256 tokens; the epoch with the best validation F1 is kept. Both transformers are trained for each of seeds 42, 43 and 44 under the dedup protocol; the seed-42 models are used for the raw split, the explanations, the robustness tests and the transfer check. RoBERTa-base seeds 43 and 44 were retrained in a separate session and their weights were not kept. We use no oversampling or text augmentation, unlike {c('uddin')}. All runs used one NVIDIA T4 GPU (PyTorch 2.11.0, Transformers 5.18.0, scikit-learn 1.6.1, SHAP 0.52.0). The whole pipeline was run twice with identical code and seeds; run 2 retained the model weights and is used for all results and controls below, and run 1 is reported as a replication in Section IV-C.")
Pp("Four controls test the comparison. (i) Tuned baselines: for each seed, LR and SVM are selected on the validation split by F1 over word n-gram range {(1,1), (1,2)} and <i>C</i> ∈ {0.1, 0.3, 1, 3, 10, 30, 100} (LR) or {0.03, 0.1, 0.3, 1, 3, 10} (SVM), and the test split is scored once. (ii) Matched input: the TF-IDF models are trained and tested on the text covered by each email’s first 256 RoBERTa tokens, which is what the transformers see. (iii) Near-duplicate-aware splits: two emails are linked when their TF-IDF cosine similarity is at least 0.95 (or 0.90); connected components stay within one partition under stratified group splitting into about 70/10/20, and the transformers were retrained on the 0.95 split for seed 42 only. (iv) Conflicting-label policies: the “empty” group is removed entirely, or one copy with its majority label (phishing) replaces the first copy.")
H2("Explainers and the LIME–IG Hybrid")
Pp(f"All explainers attribute the predicted-class probability (the phishing probability in the main sample) to words. LIME {c('ribeiro')} uses 500 perturbed samples per email. SHAP {c('lundberg')} uses the Partition explainer with a text masker and at most 300 model evaluations. IG {c('sundararajan')} uses Captum’s layer integrated gradients {c('kokhlikyan')} at the embedding layer with 30 steps and a baseline in which every non-special token is replaced by the pad token (our default); attributions are summed over embedding dimensions and over the sub-words of each word. Transformers Interpret {c('pierse')}, used in {c('uddin')}, wraps a gradient-based attribution; we call IG directly and so call the combination a LITA-style hybrid. The hybrid combines word scores <i>L<sub>i</sub></i> (LIME) and <i>T<sub>i</sub></i> (IG) as")
EQ("EQ1")
Pp(f"with <i>α</i> ∈ [0, 1]. Reference {c('uddin')} fixes <i>α</i> = 0.5 and normalises each method to [−1, 1] by min-max scaling,")
EQ("EQ2")
Pp("which shifts a zero (neutral) score to a non-zero value. We also test a sign-preserving alternative,")
EQ("EQ3")
Pp("and select <i>α</i> on validation emails by maximising mean comprehensiveness over a grid of 0, 0.1, …, 1; the tuned hybrid uses the sign-preserving normalisation. Control experiments (Section IV-D) vary IG’s target (the predicted-phishing probability or the logit difference), its number of steps (30, 100 or 300) and its baseline (pad or mask token), and add Gradient×Input. The completeness error is Captum’s convergence delta divided by |<i>F</i>(<i>x</i>) − <i>F</i>(<i>x</i><sub>0</sub>)|, summarised by its median over emails. SHAP is also run with 1,000 and 2,000 evaluations on the first 30 emails.")
H2("Faithfulness and Agreement Metrics")
Pp("Words are alphanumeric tokens, compared case-insensitively. Let <i>ŷ</i>(<i>x</i>) be the class predicted for email <i>x</i>, <i>p<sub>ŷ</sub></i>(<i>z</i>) the probability the model assigns to that class for a text <i>z</i> (the reference class stays fixed after deletion), <i>S<sub>k</sub></i> the <i>k</i> highest-scoring words supporting <i>ŷ</i>(<i>x</i>) (the most phishing-indicative words for emails predicted phishing, the most safe-indicative for emails predicted safe), <i>x</i>\\<i>S<sub>k</sub></i> the email with every occurrence of those words deleted and <i>x</i>|<i>S<sub>k</sub></i> the email reduced to those words in their original order. Following " + c('deyoung') + ",")
new.append(TNR(clone(PR["EQ4"], "\tComp(<i>k</i>) = <i>p<sub>ŷ</sub></i>(<i>x</i>) − <i>p<sub>ŷ</sub></i>(<i>x</i>\\<i>S<sub>k</sub></i>)\t(4)"))); new.append(TNR(clone(PR["EQ5"], "\tSuff(<i>k</i>) = <i>p<sub>ŷ</sub></i>(<i>x</i>) − <i>p<sub>ŷ</sub></i>(<i>x</i>|<i>S<sub>k</sub></i>)\t(5)")))
Pp(f"We report each metric averaged over <i>k</i> = 1, …, 10 and over emails; higher comprehensiveness (Comp.) and lower sufficiency (Suff.) are better. A random word ordering gives the reference level. The explained set is 100 test emails drawn at random (fixed seed) from the dedup seed-42 test emails that are truly phishing and that the model also classifies as phishing on their first 120 words, plus 40 such emails from the validation split for choosing <i>α</i>; each email is cut to its first 120 words, and the two models are explained on different emails. Methods are compared with paired two-sided Wilcoxon signed-rank tests {c('wilcoxon')} with Holm’s correction {c('holm')} over 13 comparisons per model and metric; means carry 95% bootstrap intervals over emails (2,000 resamples); p-values in the control experiments are uncorrected and exploratory. Agreement between two explainers is the Spearman correlation of their word scores, the Jaccard overlap of their five highest-scoring words, and the share of LIME’s ten largest-magnitude words whose sign differs from IG.")
Pp("Two further checks address the explanation sample. LIME is rerun with three new random seeds on the same checkpoint and the same 100 emails (IG is deterministic). A stratified sample of 120 test emails, identical for both models, holds 20 each of confident and non-confident correctly classified phishing and safe emails and 40 emails misclassified by at least one model, each cut to 120 words; an email is confident if the smaller of the two models’ probabilities for the predicted class is at least 0.99 and non-confident if it is below 0.99 (both models correct); the rule was fixed before any explanation was computed. For emails predicted safe, the most safe-indicative words are ranked first and every metric uses the probability of the predicted class. IG with a mask baseline uses the logit target and 100 steps; the hybrids use <i>α</i> = 0.5 and abs-max normalisation, fixed in advance and not tuned on this sample.")
H2("Robustness, Transfer and Statistics")
Pp(f"Three random, non-adaptive perturbations {c('boucher')} are applied to the full text of every dedup seed-42 test email, with the same perturbed text for every model: homoglyph (15% of the letters a, e, o, p, c replaced by Cyrillic look-alikes), typo (an adjacent-letter swap in 10% of words longer than three letters) and zero-width (a zero-width space inserted in 10% of such words). To separate truncation from content change, homoglyphed emails are also scored with a 512-token limit and, for each perturbation, on the subset whose perturbed text fits in 256 tokens. For a transfer check, 3,000 emails sampled at random from each of two public corpora, Enron-Spam {c('metsis','enron_hf')} (spam mapped to the positive class) and zefang-liu/phishing-email-dataset {c('zefang')}, are scored by the dedup seed-42 models. Detectors are compared on the same test emails with the exact form of McNemar’s test {c('mcnemar')} (Holm-corrected over the 24 matched-input comparisons) and with paired bootstrap intervals (1,000 resamples) for the accuracy difference. The notebooks, split definitions and result files are available from the author on request.")

# ================================================================== RESULTS
H1("Results")
H2("Duplicate Audit")
t1 = [["Item", "Value"],
      ["Emails after cleaning", f"{a['n_raw']:,}"],
      ["Duplicate groups (rows in them)", f"{n_groups} ({n_rows_groups:,}; {100*n_rows_groups/a['n_raw']:.1f}%)"],
      ["Group “empty”: copies (phishing / safe)", f"{n_empty_all} ({n_empty_phish} / {n_empty_all - n_empty_phish})"],
      ["Emails after deduplication", f"{a['n_dedup']:,}"],
      ["Raw test emails also in training", f"{n_seen}/3,727 ({100*a['raw_test_seen_in_train_frac_seed42']:.1f}%)"],
      ["…of which text is “empty”", f"{n_empty_seen}"],
      ["Acc. on those {0} (LR / Distil / RoBERTa)".format(n_seen), f"{P(float(acc_seen_lr),1)} / {P(seenT[DI].acc,1)} / {P(seenT[RO].acc,1)}"],
      [f"Acc. on the other {n_other_seen} (LR)", f"{P(float(acc_other_seen),1)}"],
      [f"Near-duplicates (cos ≥ 0.95), raw test; LR acc.", f"{n_near} ({float(pct_near):.1f}%); {P(float(acc_near),1)}"],
      ["Acc. on unseen raw test (Distil / RoBERTa)", f"{P(seenF[DI].acc,1)} / {P(seenF[RO].acc,1)}"]]
TAB("Duplicate and overlap audit of the corpus (counts; accuracy in %)", make_table(t1, [3000, 1881]))
Pp(f"Table I summarises the audit. {n_groups} groups of identical text hold {n_rows_groups:,} emails ({100*n_rows_groups/a['n_raw']:.1f}%): {hist['(1, 2]']} pairs, {hist['(2, 3]']} triples, {hist['(3, 5]']} groups of four or five, {hist['(5, 10]']} of six to ten, and one group of {n_empty_all}. That largest group is the only one with conflicting labels, and its text is the single word “empty” ({n_empty_phish} copies labelled phishing, {n_empty_all - n_empty_phish} safe). In the raw split, {n_seen} test emails ({100*a['raw_test_seen_in_train_frac_seed42']:.1f}%) also occur in training, and {n_empty_seen} of them are “empty” rows (64 phishing, 40 safe). On these {n_seen} emails DistilRoBERTa, RoBERTa-base and TF-IDF LR all reach {P(seenT[DI].acc,1)}% accuracy, against {P(seenF[DI].acc,1)}–{P(seenF[RO].acc,1)}% on unseen emails. For LR, which we examined email by email, all errors are safe-labelled “empty” rows that it classifies as phishing, the majority label of that text; the other {n_other_seen} overlapping emails, and the {n_near} test emails with a near-duplicate in training but no exact copy, are classified with {P(float(acc_other_seen),0)}% accuracy. Each transformer makes exactly {int(seenT[DI].fp)} errors on these {n_seen} emails, all false positives, which equals the number of safe-labelled “empty” rows among them; we did not verify per email that these are the same rows.")
Pp(f"The raw-split overlap analysis is therefore confounded by a conflicting-label placeholder: LR’s errors on the overlapping emails were all safe-labelled “empty” records, while the remaining overlapping emails were classified with high accuracy. Because the transformers’ errors were not verified at the individual-email level, the overall effect of duplicate overlap on measured accuracy remains unresolved. The placeholder also sets a ceiling. Whatever label a text-only classifier gives to “empty”, it errs on the minority copies, so at least 40 of the 3,727 raw test emails ({floor_pct:.2f}%) are unavoidable errors and raw accuracy cannot exceed about {100-floor_pct:.1f}%. One “empty” row survives deduplication; residual near-duplicates in the deduplicated data are examined next. Non-ASCII characters make up {100*na_phish:.2f}% of the characters of phishing emails and {100*na_safe:.2f}% of safe emails ({na_phish/na_safe:.0f} times more), and no email in the corpus contains Cyrillic.")
_f95 = _ndl[(_ndl.thr == 0.95)]; _f90 = _ndl[(_ndl.thr == 0.90)]
_fl95 = _f95.n_flagged.iloc[0]; _fl90 = _f90.n_flagged.iloc[0]
_acc_fl = min(_f95.acc_flagged.min(), _f90.acc_flagged.min()); _un = _ndl.acc_unflagged
_g95 = _ndg[(_ndg.thr == 0.95) & _ndg.model.isin(["LR", "SVM"])].acc * 100; _g90 = _ndg[(_ndg.thr == 0.90) & _ndg.model.isin(["LR", "SVM"])].acc * 100
_dd_lrsvm = _tuned[_tuned.model.isin(["LR_default", "SVM_default"])].acc * 100
_gt = _ndg[(_ndg.thr == 0.95) & _ndg.model.isin([DI, RO])].set_index("model").acc * 100
_e = _emp[_emp.protocol == "dedup"]; _e = _e[_e.model.isin(["LR", "SVM"])]
_base = _e[_e.policy == "first_copy (paper)"].set_index(["model", "seed"]).acc * 100
_d_rm = (_e[_e.policy == "group_removed"].set_index(["model", "seed"]).acc * 100 - _base)
_d_mj = (_e[_e.policy == "majority_label_rep"].set_index(["model", "seed"]).acc * 100 - _base)
_rw = _emp[_emp.protocol == "raw"].set_index(["policy", "model"]).acc * 100
Pp(f"Near-duplicates do not obviously explain the accuracy either. In the deduplicated seed-42 test set, {_fl95} emails ({100*_f95.frac_flagged.iloc[0]:.1f}%) have a training email with cosine similarity of at least 0.95 ({_fl90}, {100*_f90.frac_flagged.iloc[0]:.1f}%, at 0.90); all four models classify them with at least {_acc_fl*100:.1f}% accuracy, against {100*_un.min():.1f}–{100*_un.max():.1f}% on the other emails. Splitting by near-duplicate group ({int(_nds.loc[0.95].n_groups):,} groups at 0.95, {int(_nds.loc[0.90].n_groups):,} at 0.90) gives broadly comparable accuracy for the TF-IDF models: LR and SVM reach {rng_(_g95)}% at 0.95 and {rng_(_g90)}% at 0.90 across seeds, against {rng_(_dd_lrsvm)}% on the exact-deduplicated split (mean differences {ND_LO:+.2f} to {ND_HI:+.2f} points; the test sets differ, so this is not a controlled comparison). The transformers were run once (seed 42, 0.95 only), as a single-seed control: DistilRoBERTa {_gt[DI]:.2f}% and RoBERTa-base {_gt[RO]:.2f}%. For the conflicting “empty” group, removing it entirely changes LR and SVM accuracy on the dedup split by {_d_rm.min():+.2f} to {_d_rm.max():+.2f} points and keeping one copy with the majority label by {_d_mj.min():+.2f} to {_d_mj.max():+.2f}, whereas on the raw split removing it raises LR and SVM from {_rw[('group_kept (paper)','LR')]:.2f}% and {_rw[('group_kept (paper)','SVM')]:.2f}% to {_rw[('group_removed','LR')]:.2f}% and {_rw[('group_removed','SVM')]:.2f}%. The dedup test sets contain no “empty” row, so the transformers’ results do not change.")
H2("Detection Performance")
def pm(m, met, d=2):
    mu, s = sd(m, met); return f"{100*mu:.{d}f} ± {100*s:.{d}f}"
t2 = [["Model", "Acc.", "F1", "FPR", "FNR"]]
for lab, m in [("TF-IDF LR", "LR"), ("TF-IDF SVM", "SVM"), ("TF-IDF RF", "RF"), ("DistilRoBERTa", DI)]:
    t2.append([lab, pm(m, "acc"), pm(m, "f1"), pm(m, "fpr"), pm(m, "fnr")])
rr = N["sd"].loc[RO]
t2.append(["RoBERTa-base", pm(RO, "acc"), pm(RO, "f1"), pm(RO, "fpr"), pm(RO, "fnr")])
TAB("Detection performance on deduplicated test sets (%; mean ± sd over three seeds; FPR and FNR: false-positive and false-negative rate)", make_table(t2, [1200, 920, 920, 920, 921]))
_a = N["all"][N["all"].protocol == "dedup"]
n_ro_above = sum(RO3["acc"][i] > float(_a[(_a.seed == sd_) & (_a.model == "SVM")].acc.iloc[0]) for i, sd_ in enumerate(RO3_SEEDS))
dl, dlo, dhi = pd_(RO, "LR"); ds_, dslo, dshi = pd_(RO, "SVM"); dd1 = pd_(DI, "LR"); dd2 = pd_(DI, "SVM")
Pp(f"Table II gives the results. The class-balanced bigram TF-IDF LR and SVM reach {pm('LR','acc')[:5]}% and {pm('SVM','acc')[:5]}% accuracy, above the 97.24% and 97.90% reported in {c('uddin')} for tuned LR and SVM on default unigram features (a 64/16/20 split and different preprocessing). DistilRoBERTa ({pm(DI,'acc')[:5]}%) matches them within seed variation, and RoBERTa-base reaches {pm(RO,'acc')[:5]}% with the lowest mean false-positive rate ({pm(RO,'fpr')}%). Its seed-by-seed accuracy ({', '.join(f'{100*v:.2f}' for v in RO3['acc'])}%) is above the SVM’s in {['none','one','two','all three'][n_ro_above]} of the three seeds, and its mean is only about {round(100*(rr.acc-N['sd'].loc['SVM'].acc),1):.1f} points higher; its false-negative rate ({pm(RO,'fnr')}%) is the highest of the four non-forest models. Only the random forest is clearly worse. McNemar tests on the same 3,499 test emails give p = {mc.loc['LR'].p_value:.2f} against LR ({int(mc.loc['LR'].a_wrong_b_right)} versus {int(mc.loc['LR'].a_right_b_wrong)} discordant emails), p = {mc.loc['SVM'].p_value:.2f} against SVM and p < 0.001 against RF. For the seed-42 models, paired bootstrap differences in accuracy are {dl:+.2f} points (95% interval {dlo:+.2f} to {dhi:+.2f}) for RoBERTa-base minus LR and {ds_:+.2f} ({dslo:+.2f} to {dshi:+.2f}) minus SVM, and {dd1[0]:+.2f} ({dd1[1]:+.2f} to {dd1[2]:+.2f}) and {dd2[0]:+.2f} ({dd2[1]:+.2f} to {dd2[2]:+.2f}) for DistilRoBERTa minus LR and SVM. All intervals contain zero but extend beyond ±0.5 points, so we found no significant difference and cannot claim equivalence within that margin.")
t7 = [["TF-IDF input; settings", "LR", "SVM", "Distil.", "RoBERTa"],
      ["Full text; fixed", fm(msd(_tuned, "LR_default")), fm(msd(_tuned, "SVM_default")), pm(DI, "acc"), pm(RO, "acc")],
      ["Full text; tuned", fm(msd(_tuned, "LR_tuned")), fm(msd(_tuned, "SVM_tuned")), "–", "–"],
      ["First 256 tokens; fixed", fm(msd(_match, "LR_default_trunc256")), fm(msd(_match, "SVM_default_trunc256")), "–", "–"],
      ["First 256 tokens; tuned", fm(msd(_match, "LR_tuned_trunc256")), fm(msd(_match, "SVM_tuned_trunc256")), "–", "–"]]
for thr in (0.95, 0.90):
    gg = _ndg[_ndg.thr == thr]
    tr_ = [f"{100*gg[(gg.model == m) & (gg.seed == 42)].acc.iloc[0]:.2f}†" if thr == 0.95 else "–" for m in (DI, RO)]
    t7.append([f"Group split, cos ≥ {thr:.2f}", fm(msd(gg, "LR")), fm(msd(gg, "SVM")), *tr_])
TAB("Accuracy (%) of the baseline comparison under different controls (mean ± sd over seeds 42–44; the transformers always see the first 256 tokens; † seed 42 only)", make_table(t7, [1420, 880, 880, 850, 851]))
_mr = _mp[_mp.transformer == RO]; _md = _mp[_mp.transformer == DI]
_drop = {m: msd(_tuned, m + "_default")[0] - msd(_match, m + "_default_trunc256")[0] for m in ("LR", "SVM")}
_tun = {m: msd(_tuned, m + "_tuned")[0] - msd(_tuned, m + "_default")[0] for m in ("LR", "SVM")}
Pp(f"Table III tests whether this comparison depends on how the baselines are built. Tuning on the validation split did not raise test accuracy (LR {_tun['LR']:+.2f}, SVM {_tun['SVM']:+.2f} points on average). Restricting the TF-IDF models to the transformers’ first 256 tokens lowers them by {_drop['LR']:.2f} (LR) and {_drop['SVM']:.2f} (SVM) points, and then RoBERTa-base is ahead of the four matched baselines in {int((_mr.diff_pts > 0).sum())} of {len(_mr)} seed–baseline pairs ({_mr.diff_pts.min():+.2f} to {_mr.diff_pts.max():+.2f} points) with an exact McNemar <i>p</i> below 0.05 in {int((_mr.mcnemar_p < .05).sum())} of them ({int((_mr.adj < .05).sum())} after Holm correction over all {len(_mp)} comparisons); DistilRoBERTa ranges from {_md.diff_pts.min():+.2f} to {_md.diff_pts.max():+.2f} points ({int((_md.mcnemar_p < .05).sum())} nominal and {int((_md.adj < .05).sum())} adjusted below 0.05). The larger transformer’s advantage is therefore small and appears only when the linear models are denied the rest of the email; we treat the tests as exploratory.")
H2("Explanation Faithfulness")
def cell_c(m, k, best=False):
    lo, hi = F[(m, "comp", "ci")][k]; return f"{comp(m,k):.3f} ({lo:.3f}–{hi:.3f})"
names = [("LIME", "LIME"), ("SHAP", "SHAP"), ("IG", "IG"), ("LITA-minmax(a=.5)", "Hybrid, α = 0.5, min-max"), ("LITA-absmax(a=.5)", "Hybrid, α = 0.5, abs-max"),
         ("LITA-absmax(tuned)", "Hybrid, tuned α*, abs-max"), ("Random", "Random order")]
t3 = [["Method", "DistilRoBERTa", "", "RoBERTa-base", ""], ["", "Comp. ↑ (95% CI)", "Suff. ↓", "Comp. ↑ (95% CI)", "Suff. ↓"]]
bold = set()
nonrand = [k for k, _ in names if k != "Random"]
for ci_, m in ((1, DI), (3, RO)):
    best_c = max(nonrand, key=lambda k: round(comp(m, k), 3)); best_s = min(nonrand, key=lambda k: round(suff(m, k), 3))
    for ri, (k, _) in enumerate(names, start=2):
        if round(comp(m, k), 3) == round(comp(m, best_c), 3) and k != "Random": bold.add((ri, ci_))
        if round(suff(m, k), 3) == round(suff(m, best_s), 3) and k != "Random": bold.add((ri, ci_ + 1))
for k, lab in names:
    t3.append([lab, cell_c(DI, k), f"{suff(DI,k):.3f}", cell_c(RO, k), f"{suff(RO,k):.3f}"])
TAB(f"Faithfulness of word-level explanations (run 2, seed-42 models; mean over 100 emails; bold = best non-random method; SHAP: 300 evaluations; IG: probability target, 30 steps, pad baseline; tuned = validation-selected LIME weight, {alpha2[DI]:.1f} for DistilRoBERTa and {alpha2[RO]:.1f} for RoBERTa-base, where it equals LIME)",
    make_table(t3, [1090, 1410, 485, 1410, 485], header_rows=2, bold_cells=bold, spans={(0, 1): 2, (0, 3): 2}))
pD = lambda met, x, y: F[(DI, met, "p")][(x, y)][1]
pR = lambda met, x, y: F[(RO, met, "p")][(x, y)][1]
HY, HT = "LITA-absmax(a=.5)", "LITA-absmax(tuned)"
Pp(f"Table IV and Fig. 2 report faithfulness. For DistilRoBERTa the tuned hybrid has the highest comprehensiveness ({comp(DI,HT):.3f}) and is better than LIME ({comp(DI,'LIME'):.3f}; Holm-adjusted {pv(pD('comp',HT,'LIME'))}), SHAP ({comp(DI,'SHAP'):.3f}) and IG ({comp(DI,'IG'):.3f}; both {pv(max(pD('comp',HT,'IG'), pD('comp',HT,'SHAP')))}). The fixed <i>α</i> = 0.5 hybrid ({comp(DI,HY):.3f}) is numerically ahead of LIME but not significantly ({pv(pD('comp',HY,'LIME'))}). All methods beat random order ({comp(DI,'Random'):.4f}; adjusted {pv(max(pD('comp',k,'Random') for k in ('LIME','SHAP','IG')))}), and sufficiency is low for all of them ({suff(DI,HY):.3f}–{suff(DI,'SHAP'):.3f}) against {suff(DI,'Random'):.3f} for random order. The validation sweep selected <i>α</i>* = {alpha2[DI]:.1f}.")
Pp(f"For RoBERTa-base, LIME ({comp(RO,'LIME'):.3f}) is the most faithful by comprehensiveness; it is better than SHAP ({comp(RO,'SHAP'):.3f}; {pv(pR('comp','LIME','SHAP'))}), and the fixed hybrid ({comp(RO,HY):.3f}) is significantly worse than LIME ({pv(pR('comp',HY,'LIME'))}). Validation selects <i>α</i>* = 1, i.e. LIME only. IG ({comp(RO,'IG'):.4f}, interval {F[(RO,'comp','ci')]['IG'][0]:.4f}–{F[(RO,'comp','ci')]['IG'][1]:.4f}) is not distinguishable from random order ({comp(RO,'Random'):.4f}; adjusted {pv(F[(RO,'comp','p')][('IG','Random')][1])}) and has sufficiency {suff(RO,'IG'):.3f} against {suff(RO,'Random'):.3f}; SHAP has the best sufficiency ({suff(RO,'SHAP'):.3f}). Absolute effects are small: Table IV averages over <i>k</i> = 1–10, and even the best method lowers the probability by only about {max(json.load(open(DATA+'/controls_ctl/curves_export.json'))[f'{RO}|LIME']['rem_mean']):.2f} at <i>k</i> = 10 (Fig. 2). This may reflect redundant cues or predictions saturated near 1; we did not test either. Section IV-D shows that the IG result depends on its configuration.")
FIG(f"{FIGDIR}/fig2_curves.png", 3.4, "Comprehensiveness (top) and sufficiency (bottom) against the number of top-ranked words <i>k</i> for DistilRoBERTa (left) and RoBERTa-base (right), averaged over 100 emails (run 2); bands are 95% normal intervals.")
Pp(f"Run 1 and run 2 used identical code and seeds, but the 100 explained emails are selected per model from its own predictions, so they differ. Detection accuracy agreed within {100*max_acc_diff:.2f} points (all runs, protocols and models), yet LIME’s comprehensiveness for DistilRoBERTa fell from {c1(DI,'LIME'):.3f} to {c2(DI,'LIME'):.3f}, and across LIME, SHAP, IG and the fixed hybrid by up to {ratio_max:.1f} times; the selected <i>α</i>* moved from {alpha1[DI]:.1f} to {alpha2[DI]:.1f}; and two uncorrected p-values changed side of 0.05 (tuned hybrid against LIME on DistilRoBERTa, 0.116 in run 1 and 0.003 in run 2; tuned hybrid against SHAP on RoBERTa-base, 0.124 and 0.006). The qualitative picture is stable: every method beats random order on DistilRoBERTa, IG is at or near random order on RoBERTa-base, and LIME is the most faithful single explainer there. Absolute values and small between-method gaps are not stable.")
H2("Sensitivity of IG and SHAP")
def tdlt(m, v):
    x = ig(m, v).get("delta_med"); return "–" if x is None else f"{x:.3f}"
vars_ = [(V_DEF, "IG, prob., 30 steps, pad (default)"), (V_P300, "IG, prob., 300 steps, pad"), (V_L100, "IG, logit, 100 steps, pad"),
         (V_MASK, "IG, logit, 100 steps, mask"), (V_GXI, "Gradient×Input, logit")]
t5 = [["Variant", "Median δ", "", "ρ with LIME", "", "Hybrid Comp.", ""], ["", "D", "R", "D", "R", "D", "R"]]
for v, lab in vars_:
    t5.append([lab, tdlt(DI, v), tdlt(RO, v), f"{ig(DI,v)['rho']:.2f}", f"{ig(RO,v)['rho']:.2f}", f"{ig(DI,v)['h_comp']:.3f}", f"{ig(RO,v)['h_comp']:.3f}"])
t5.append(["LIME alone (reference)", "", "", "", "", f"{comp(DI,'LIME'):.3f}", f"{comp(RO,'LIME'):.3f}"])
TAB("IG settings (D: DistilRoBERTa, R: RoBERTa-base; same 100 emails as Table IV; δ: relative completeness error; hybrid: alpha = 0.5, abs-max, with that IG variant; LIME row: Comp. of LIME itself)",
    make_table(t5, [1350, 420, 420, 420, 420, 425, 426], header_rows=2, spans={(0, 1): 2, (0, 3): 2, (0, 5): 2}))
mk_ro, mk_di = ig(RO, V_MASK), ig(DI, V_MASK)
Pp(f"Fig. 3 and Table V vary the IG configuration. For RoBERTa-base, raising IG’s steps from 30 to 300 does not change the picture (comprehensiveness {ig(RO,V_DEF)['comp']:.4f} to {ig(RO,V_P300)['comp']:.4f}; {pv(ig(RO,V_P300)['p_rand'])} against random order), although the median completeness error is already {ig(RO,V_DEF)['delta_med']:.3f} at 30 steps, so the weak result is not a convergence failure. Switching the target to the logit gives {ig(RO,V_L100)['comp']:.4f} ({pv(ig(RO,V_L100)['p_rand'])}). Replacing the pad baseline by the mask token, however, raises comprehensiveness to {mk_ro['comp']:.3f} (interval {mk_ro['ci'][0]:.3f}–{mk_ro['ci'][1]:.3f}), far above random order ({pv(mk_ro['p_rand'])}) and indistinguishable from LIME ({pv(mk_ro['p_lime'])}); its correlation with LIME rises from {ig(RO,V_DEF)['rho']:.2f} to {mk_ro['rho']:.2f}. The IG result for RoBERTa-base in Table IV is therefore a property of the pad baseline, not of the model. For DistilRoBERTa the logit target alone lifts IG from {ig(DI,V_DEF)['comp']:.3f} to {ig(DI,V_L100)['comp']:.3f}, above LIME ({comp(DI,'LIME'):.3f}; {pv(ig(DI,V_L100)['p_lime'])}), and the mask baseline gives {mk_di['comp']:.3f} with a LIME correlation of {mk_di['rho']:.2f}.")
FIG(f"{FIGDIR}/fig3_ig_controls.png", 3.4, "Comprehensiveness (mean over <i>k</i> = 1–10, 95% bootstrap interval) of LIME, random order and IG variants on the same 100 emails per model.")
hy = lambda m, v: ig(m, v)
Pp(f"These settings also decide whether the hybrid helps. With the default IG the α = 0.5 hybrid is not significantly different from LIME on DistilRoBERTa ({hy(DI,V_DEF)['h_comp']:.3f} against {comp(DI,'LIME'):.3f}; {pv(hy(DI,V_DEF)['h_p_lime'])}) and worse on RoBERTa-base ({hy(RO,V_DEF)['h_comp']:.3f} against {comp(RO,'LIME'):.3f}). With the mask baseline it is better than LIME on both models ({hy(DI,V_MASK)['h_comp']:.3f} against {comp(DI,'LIME'):.3f} and {hy(RO,V_MASK)['h_comp']:.3f} against {comp(RO,'LIME'):.3f}; both {pv(max(hy(DI,V_MASK)['h_p_lime'], hy(RO,V_MASK)['h_p_lime']))}, uncorrected). The margins over LIME are small (0.012 and 0.008 in comprehensiveness).")
s30d = F[(DI, "first30")]; s30r = F[(RO, "first30")]
Pp(f"SHAP’s budget matters for DistilRoBERTa. On the first 30 emails its comprehensiveness rises from {shr(DI,300).comp:.4f} at 300 evaluations to {shr(DI,1000).comp:.4f} at 1,000 and {shr(DI,2000).comp:.4f} at 2,000, against {s30d['LIME']:.4f} for LIME on the same emails, while its sufficiency falls from {shr(DI,300).suff:.4f} to {shr(DI,2000).suff:.4f}; the cost grows from {shr(DI,300).sec_per_email:.1f} to {shr(DI,2000).sec_per_email:.1f} s per email. For RoBERTa-base, SHAP is stable ({shr(RO,300).comp:.4f}, {shr(RO,1000).comp:.4f}, {shr(RO,2000).comp:.4f}; LIME {s30r['LIME']:.4f}). The low SHAP score for DistilRoBERTa in Table IV reflects the 300-evaluation budget.")
H2("Repeated LIME and a Shared Stratified Sample")
_lrm = _lr.groupby("model")
_rho = _lr[_lr.repeat > 0].groupby("model").rho_vs_run0.mean()
_hm = _lr.groupby("model").hyb_mask_minus_lime.agg(["min", "max"]); _hp = _lr.groupby("model").hyb_pad_minus_lime.agg(["min", "max"])
_ig_dev = float(pd.read_csv(f"{R2D}/lime_repeat_per_email.csv").ig_maxdiff.max())
Pp(f"Repeating LIME with three new seeds on the same checkpoint and emails changes its mean comprehensiveness only moderately ({_lrs.loc[DI,'min']:.3f}–{_lrs.loc[DI,'max']:.3f} for DistilRoBERTa, {_lrs.loc[RO,'min']:.3f}–{_lrs.loc[RO,'max']:.3f} for RoBERTa-base; at most {LIME_RATIO:.1f}-fold), although its word rankings agree only partly between seeds (mean Spearman ρ with the first run {_rho[DI]:.2f} and {_rho[RO]:.2f}). IG’s attributions were identical across repeats up to floating-point error (largest difference {_ig_dev:.0e}). The hybrid with the mask-baseline IG exceeds LIME in all four runs on both models (DistilRoBERTa {_hm.loc[DI,'min']:+.3f} to {_hm.loc[DI,'max']:+.3f}, RoBERTa-base {_hm.loc[RO,'min']:+.3f} to {_hm.loc[RO,'max']:+.3f}), whereas with the pad-baseline IG the difference is {_hp.loc[DI,'min']:+.3f} to {_hp.loc[DI,'max']:+.3f} and {_hp.loc[RO,'min']:+.3f} to {_hp.loc[RO,'max']:+.3f}. Re-sampling LIME therefore cannot explain the {ratio_max:.1f}-fold difference between the two pipeline runs (Section IV-C); it must come from the retrained model or the sampled emails, which that design does not separate.")
_lab = [("LIME", "LIME"), ("SHAP", "SHAP"), ("IG_pad", "IG, pad baseline"), ("IG_mask", "IG, mask baseline"), ("Hybrid_pad", "Hybrid, pad IG"), ("Hybrid_mask", "Hybrid, mask IG"), ("Random", "Random order")]
def _sm(m, mk, st=None):
    d = _st[(_st.model == m) & (_st.method == mk)]
    if st: d = d[d.stratum == st]
    return d.comp.mean()
_cols = [None, "phishing_ok_confident", "phishing_ok_borderline", "safe_ok_confident", "safe_ok_borderline", "model_error"]
t8 = [["Method", "All", "Phishing", "", "Safe", "", "Errors"], ["", "", "Conf.", "Non-conf.", "Conf.", "Non-conf.", ""]]
for m_, nm_ in ((DI, "DistilRoBERTa"), (RO, "RoBERTa-base")):
    t8.append([nm_, "", "", "", "", "", ""])
    for mk, lb in _lab:
        t8.append([lb] + [f"{_sm(m_, mk, st_):.3f}" for st_ in _cols])
TAB("Comprehensiveness on the shared stratified sample (seed-42 models; all = all 120 emails; Phishing and Safe = emails of that true class that both models classify correctly, 20 per cell, split into confident (smaller predicted-class probability at least 0.99) and non-confident; Errors = the 40 emails misclassified by at least one model)",
    make_table(t8, [1130, 625, 625, 625, 625, 625, 625], header_rows=2, bold_cells={(2, 0), (10, 0)}, spans={(0, 2): 2, (0, 4): 2, (2, 0): 7, (10, 0): 7}))
def _pair(m, a_, b_):
    d = _st[_st.model == m].pivot_table(index="email", columns="method", values="comp"); x = (d[a_] - d[b_]).values
    lo, hi = boot_mean(x); return x.mean(), lo, hi
_bd = _st[(_st.stratum.isin(["phishing_ok_borderline", "model_error"]))]
_sat = _st[_st.stratum == "phishing_ok_confident"].groupby(["model", "method"]).comp.mean()
hm_d, hm_dl, hm_dh = _pair(DI, "Hybrid_mask", "LIME"); hm_r, hm_rl, hm_rh = _pair(RO, "Hybrid_mask", "LIME")
hp_r = _pair(RO, "Hybrid_pad", "LIME"); ip_r = _pair(RO, "IG_pad", "Random")
_ph_b = _st[_st.stratum == "phishing_ok_borderline"].groupby(["model", "method"]).comp.mean()
_sc = _st[_st.stratum == "safe_ok_confident"].groupby(["model", "method"]).comp.mean()
_sb = _st[_st.stratum == "safe_ok_borderline"].groupby(["model", "method"]).comp.mean()
Pp(f"Table VI reports the same 120 emails explained for both models, split into the five strata of Section III-D. Comprehensiveness is near zero for confidently classified phishing emails (at most {_sat.max():.3f} for any method), because deleting words barely moves a probability close to 1; it is not near zero for confidently classified safe emails (LIME {_sc[(DI,'LIME')]:.2f} and {_sc[(RO,'LIME')]:.2f}), and it is larger for non-confident emails and for errors (LIME on non-confident phishing emails {_ph_b[(DI,'LIME')]:.2f} for DistilRoBERTa and {_ph_b[(RO,'LIME')]:.2f} for RoBERTa-base, on non-confident safe emails {_sb[(DI,'LIME')]:.2f} and {_sb[(RO,'LIME')]:.2f}). Averaged over the sample, LIME is more faithful than SHAP at 300 evaluations for both models, and the mask-baseline IG is far ahead of the pad-baseline IG on RoBERTa-base ({_sm(RO,'IG_mask'):.3f} against {_sm(RO,'IG_pad'):.3f}), which in this sample is above random order ({ip_r[0]:+.3f}, 95% interval {ip_r[1]:+.3f} to {ip_r[2]:+.3f}) although far below LIME ({_sm(RO,'LIME'):.3f}). The hybrid with the mask-baseline IG exceeds LIME by {hm_d:+.3f} ({hm_dl:+.3f} to {hm_dh:+.3f}) on DistilRoBERTa but by only {hm_r:+.3f} ({hm_rl:+.3f} to {hm_rh:+.3f}) on RoBERTa-base, where the hybrid with the pad-baseline IG is worse than LIME ({hp_r[0]:+.3f}, {hp_r[1]:+.3f} to {hp_r[2]:+.3f}). These paired differences are exploratory and use bootstrap intervals over emails.")
H2("Agreement Between Explainers")
Pp(f"Agreement with the default IG is weak (seed-42 models, means over 100 emails). The LIME–IG Spearman correlation is <i>ρ</i> = {ag.loc[DI,'spearman_lime_ig']:.2f} for DistilRoBERTa and {ag.loc[RO,'spearman_lime_ig']:.2f} for RoBERTa-base, where unrelated rankings give <i>ρ</i> near 0; LIME and IG disagree on the sign of {100*ag.loc[DI,'sign_disagree_top10']:.0f}% and {100*ag.loc[RO,'sign_disagree_top10']:.0f}% of LIME’s ten largest-magnitude words, the latter about what chance would give; and the top-5 Jaccard overlap is {ag.loc[DI,'top5_jaccard']:.2f} and {ag.loc[RO,'top5_jaccard']:.2f}, against roughly 0.03–0.04 for random five-word sets drawn from 60–100 distinct words. Much of this is a configuration effect: with the mask baseline the LIME–IG correlation is {mk_di['rho']:.2f} and {mk_ro['rho']:.2f}. LIME and SHAP agree only modestly ({ag.loc[DI,'spearman_lime_shap']:.2f} and {ag.loc[RO,'spearman_lime_shap']:.2f}), so explainers can point an analyst to different words for the same prediction.")
H2("Robustness and Transfer")
fh_lr = rb("TFIDF-LR", "homoglyph")
def df1(m): return 100 * (rb(m, "none").f1 - rb(m, "homoglyph").f1)
Pp(f"Fig. 4(a) shows the false-positive rate under perturbation for the seed-42 models, so the unperturbed values differ from the three-seed means in Table II. Homoglyph substitution lowers F1 by {df1(DI):.2f} points for DistilRoBERTa, {df1(RO):.2f} for RoBERTa-base and {df1('TFIDF-LR'):.2f} for TF-IDF LR. For the transformers the loss comes from new false alarms: the false-positive rate rises from {P(rb(DI,'none').fpr)}% to {P(rb(DI,'homoglyph').fpr)}% for DistilRoBERTa ({int(rb(DI,'none').fp)} to {int(rb(DI,'homoglyph').fp)} emails) and from {P(rb(RO,'none').fpr)}% to {P(rb(RO,'homoglyph').fpr)}% for RoBERTa-base, whereas LR’s moves from {P(rb('TFIDF-LR','none').fpr)}% to {P(fh_lr.fpr)}%. Truncation does not explain this. Homoglyphs lengthen the token sequence (mean {tokm:.0f} tokens clean, {rcr(DI,'homoglyph').mean_tokens:.0f} perturbed, and {100*fr256:.0f}% of clean emails already exceed 256 tokens), but with a 512-token limit the false-positive rate is still {P(rcr(DI,'homoglyph',512).fpr)}% and {P(rcr(RO,'homoglyph',512).fpr)}%, and on the {int(rcr(DI,'homoglyph_fits256_only').n):,} emails that fit in 256 tokens even after perturbation it rises from {P(rcr(DI,'homoglyph_fits256_only').fpr_clean_same_subset)}% to {P(rcr(DI,'homoglyph_fits256_only').fpr)}% (DistilRoBERTa) and from {P(rcr(RO,'homoglyph_fits256_only').fpr_clean_same_subset)}% to {P(rcr(RO,'homoglyph_fits256_only').fpr)}% (RoBERTa-base). A possible cause is that non-ASCII characters are {na_phish/na_safe:.0f} times more frequent in the phishing class and Cyrillic never occurs in training, so any non-ASCII text may read as phishing; we did not test this. Typo and zero-width perturbations changed F1 by at most {100*max(abs(rb(m,'none').f1-rb(m,a_).f1) for m in ['TFIDF-LR',DI,RO] for a_ in ('typo','zero_width')):.2f} points for every model.")
FIG(f"{FIGDIR}/fig4_robustness.png", 3.4, "False-positive rate under perturbation (dedup, seed 42, single run per model): (a) all test emails; (b) homoglyph truncation controls for the transformers.")
enr_non = int(enr.n_nonoverlap)
Pp(f"On 3,000 Enron-Spam emails the dedup seed-42 models reached {P(xd.loc[('SetFit/enron_spam',DI),'acc'])}% (DistilRoBERTa), {P(xd.loc[('SetFit/enron_spam',RO),'acc'])}% (RoBERTa-base) and {P(xd.loc[('SetFit/enron_spam','TFIDF-LR'),'acc'])}% (TF-IDF LR) accuracy. However, {int(enr.in_kaggle_any):,} of these emails ({100*enr.frac_in_kaggle:.1f}%) also occur in the Kaggle corpus ({int(enr.in_train_seed42)} in the seed-42 training split). On the {enr_non:,} non-overlapping emails DistilRoBERTa reaches {P(enr['acc_nonoverlap_distilroberta-base'])}% and RoBERTa-base {P(enr['acc_nonoverlap_roberta-base'])}%; TF-IDF LR reaches {100*_extc.loc[('SetFit/enron_spam','LR'),'acc_nonoverlap']:.2f}% and the TF-IDF SVM {100*_extc.loc[('SetFit/enron_spam','SVM'),'acc_nonoverlap']:.2f}% ({100*_extc.loc[('SetFit/enron_spam','SVM'),'acc_all']:.2f}% on all 3,000). This is weak evidence of transfer: it concerns spam, not phishing, and the sample is small. The second corpus, zefang-liu/phishing-email-dataset, is a near-complete mirror of the Kaggle corpus, not an independent test: {int(zef.in_kaggle_any):,} of its 3,000 sampled emails ({100*zef.frac_in_kaggle:.1f}%) occur in the Kaggle corpus ({int(zef.in_train_seed42):,} of them in the seed-42 training split) and only {int(zef.n_nonoverlap)} do not. We therefore do not report its accuracies as transfer.")

# ================================================================== DISCUSSION
H1("Discussion")
Pp(f"<b>Benchmark saturation.</b> On this corpus we found no significant accuracy difference between fine-tuned transformers and linear models on full text. The audit found one ambiguous placeholder that lowers raw-split accuracy and confounds the exact-duplicate overlap analysis (the overall effect of duplicate overlap on measured accuracy therefore remains unresolved), near-duplicates that are easy to classify, and a second public dataset that mirrors the corpus almost completely. When the linear models see only the transformers’ 256 tokens, RoBERTa-base is ahead in every seed–baseline pair, by up to about one point, but only {int((_mr.adj < .05).sum())} of {len(_mr)} paired tests remain significant after Holm correction, so any transformer advantage here is small and depends on the input policy. Reports of such an advantage should include strong baselines on matched input, paired confidence intervals and a statement of how duplicates and overlapping external data were handled. The result does not show that transformers are no better in general. Likewise, near-duplicate-aware group splitting produced broadly comparable TF-IDF accuracy in the tested protocols, but because the test sets differ and the transformer group-split results cover one seed, these experiments do not establish that near-duplicate overlap has no effect on reported accuracy.")
Pp(f"<b>Explanation claims need configuration checks.</b> The same explainer family gave opposite conclusions under different settings (IG against random order on RoBERTa-base with a pad or a mask baseline; SHAP at 300 or 2,000 evaluations), and DistilRoBERTa’s comprehensiveness changed by up to {ratio_max:.1f} times between two pipeline runs, although re-sampling LIME alone moved it by at most {LIME_RATIO:.1f} times. This agrees with the warning of Bastings et al. {c('bastings')} that a configuration should be validated for each task and model. A comparison of explainers should report a random-order reference, at least two IG baselines and targets with a completeness check, a budget sweep for sampling-based methods, and a repeated run.")
Pp(f"<b>Relation to the base work.</b> We do not dispute that LITA can make explanations easier to read. With a mask-baseline IG the LIME–IG hybrid of the form used in {c('uddin')} beat LIME on DistilRoBERTa in both explanation samples, but on RoBERTa-base only slightly in the first and not distinguishably in the shared sample; with the pad baseline it was worse there. Faithfulness was not measured in {c('uddin')}. Our setup is not an exact replication (shorter inputs, fewer epochs, class weights instead of augmentation, direct IG), and our RoBERTa-base raw-split accuracy ({P(r42[RO].acc)}%) is below the 98.45% reported there.")

# ================================================================== LIMITATIONS
H1("Limitations")
Pp(f"Results come from a single corpus of undocumented provenance. The main explanation sample is 100 confidently flagged phishing emails per model, cut to 120 words and different for the two models; the shared stratified sample holds only 20 emails per ordinary stratum, and deleting ten words changes confidently classified phishing predictions little, a low-contrast test. Single-run LIME scores are noisy (mean Spearman <i>ρ</i> between seeds {_rho[DI]:.2f} and {_rho[RO]:.2f}). Deletion favours perturbation-based explainers, creates off-distribution inputs and measures fidelity to the model rather than usefulness to people; we ran no user study. The IG and SHAP controls are exploratory, use uncorrected p-values and few settings (SHAP only 30 emails), and the hybrid’s <i>α</i> was selected on 40 emails. Detection uses three seeds per model, but the paired tests, explanations, robustness tests and transfer check use the seed-42 models only, and only run 2’s weights are available. Near-duplicate-aware splits were run for the linear models over three seeds but for the transformers once (seed 42, threshold 0.95), so that result is a single-seed control; the explanation, robustness and transfer experiments use the standard dedup split; one “empty” row remains after deduplication; and {100*fr256:.0f}% of test emails exceed the 256-token limit, which the matched-input control addresses only for the linear models. The perturbations are random, non-adaptive and applied to every email, and the link between non-ASCII frequency and the homoglyph false positives is a hypothesis. Enron-Spam measures spam rather than phishing and overlaps the training corpus by {100*enr.frac_in_kaggle:.0f}%.")

# ================================================================== CONCLUSION
H1("Conclusion")
Pp(f"Auditing a public phishing-email benchmark showed that its only label-conflicting duplicate group is a placeholder text that caps raw-split accuracy, that the raw-split exact-duplicate overlap is confounded by that placeholder, so its effect on measured accuracy remains unresolved, that near-duplicate-aware splits changed TF-IDF accuracy by at most {ND_ABS:.2f} points on different test sets (transformers: one seed), which does not establish that near-duplicate overlap has no effect, that no significant accuracy difference separates linear baselines from fine-tuned transformers on full text although RoBERTa-base is modestly ahead when the baselines see the same 256 tokens, and that a second public dataset is a near-complete mirror of the first. The faithfulness of word-level explanations depended on configuration and on the sampled emails: IG scored like random order on RoBERTa-base with a pad baseline in the main sample yet matched LIME with a mask baseline, SHAP’s score depended on its budget, and a LIME–IG hybrid beat LIME on DistilRoBERTa but not on RoBERTa-base in a shared stratified sample, in exploratory tests. We recommend deduplicating before splitting, reporting strong baselines on matched input with paired intervals, checking overlap of external corpora, and validating each explanation method with deletion metrics, a random reference, alternative settings and a repeated run for every model before deployment. Future work should add user studies, adaptive attacks, multi-seed transformer controls and corpora with documented provenance.")

# ================================================================== REFERENCES
_h5 = copy.deepcopy(PR["H5"]); _pp = _h5.find(qn("w:pPr"))
if _pp is not None and _pp.find(qn("w:keepNext")) is None: _pp.insert(1, OxmlElement("w:keepNext"))
new.append(_h5)
for k in ORDER:
    e = clone(PR["REF"], REFS[k])
    _ppr = e.find(qn("w:pPr")); _jc = _ppr.find(qn("w:jc"))
    if _jc is None:
        _jc = OxmlElement("w:jc"); _ppr.append(_jc)
    _jc.set(qn("w:val"), "left")           # ragged right: avoids stretched URL/DOI lines
    new.append(e)
unused = set(REFS) - set(ORDER)
assert not unused, f"uncited references: {unused}"

for e in new: tail.addprevious(e)
doc.save(OUT)
print("saved", OUT, "| references:", len(ORDER), "| abstract words:", len(ABSTRACT.split()))
