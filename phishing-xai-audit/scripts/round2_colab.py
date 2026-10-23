# =====================================================================================
# round2_colab.py -- second round of experiments for the phishing-XAI paper (Colab T4)
# Answers the reviewer-style criticisms that can be tested with new runs.
#
# Run AFTER the main notebook and controls_colab.py (it reads their Drive outputs/models).
#   !pip -q install transformers captum lime shap scikit-learn scipy datasets
#   !python round2_colab.py --drive /content/drive/MyDrive/phish_xai --csv /content/drive/MyDrive/phish_xai/Phishing_Email.csv --steps tuned,matched,neardup,empty,enron_lr
#
# Steps (select with --steps):
#   tuned       validation-tuned TF-IDF LR/SVM baselines (documented grid), 3 seeds           CPU  ~5 min
#   matched     TF-IDF baselines on the SAME first-256-token text the transformers see        CPU  ~5 min
#   neardup     near-duplicate leakage in the dedup split + group-aware splits (cos>=0.95/0.90) CPU ~10 min
#               (add --nd-transformers distilroberta-base,roberta-base to also train them: GPU, ~50 min for seed 42)
#   empty       sensitivity of the results to the conflicting "empty" group                    CPU  ~3 min
#   enron_lr    TF-IDF LR/SVM on the Enron-Spam sample, incl. the non-overlapping subset       CPU  ~2 min
#   lime_repeat repeat LIME with new seeds on the SAME checkpoints and the SAME 100 emails     GPU  ~15 min
#   strat       paired, stratified explanation sample (phishing+safe, confident+borderline+errors) GPU ~25 min
#   reseeds     OPTIONAL: retrain RoBERTa-base seeds 43,44 with eval-mode validation          GPU  ~70 min
# Output: <drive>/full/round2/*.csv and round2_results.zip  -> send the zip back.
# Every step checkpoints into <drive>/full/ckpt/ or <drive>/full/round2/ and can be re-run.
# =====================================================================================
import argparse, ast, json, math, os, pickle, random, re, sys, time, zipfile
import numpy as np, pandas as pd
from scipy import stats, sparse

ap = argparse.ArgumentParser()
ap.add_argument("--drive", default="/content/drive/MyDrive/phish_xai")
ap.add_argument("--csv", default=None)
ap.add_argument("--steps", default="tuned,matched,neardup,empty,enron_lr")
ap.add_argument("--models", default="distilroberta-base,roberta-base")
ap.add_argument("--seeds", default="42,43,44", help="seeds for the classical baselines")
ap.add_argument("--nd-transformers", default="", help="comma list of transformers to train on the group-aware split")
ap.add_argument("--nd-seeds", default="42")
ap.add_argument("--nd-tf-thr", type=float, default=0.95)
ap.add_argument("--n-lime-repeats", type=int, default=3)
ap.add_argument("--n-expl", type=int, default=100, help="emails for lime_repeat (100 = same as the paper)")
ap.add_argument("--with-rf", action="store_true", help="also fit the 300-tree random forest in the classical steps (slow)")
ap.add_argument("--n-strat", type=int, default=40, help="emails per ordinary stratum for 'strat' (error stratum: up to n-strat)")
a = ap.parse_args()
STEPS = set(a.steps.split(",")); MODELS = a.models.split(","); SEEDS = [int(s) for s in a.seeds.split(",")]
FULL = os.path.join(a.drive, "full"); CKPT = os.path.join(FULL, "ckpt"); RES = os.path.join(FULL, "results")
OUT = os.path.join(FULL, "round2"); os.makedirs(OUT, exist_ok=True)
CSV = a.csv or os.path.join(a.drive, "Phishing_Email.csv")
def ckf(*p): return os.path.join(CKPT, "_".join(str(x) for x in p))
def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False); print(f"[saved] {name}  ({len(df)} rows)")

# ---------------- helpers copied from the main notebook / controls_colab.py (same definitions) ----------------
WORD = re.compile(r"\w+")
def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t)); return re.sub(r"\s+", " ", t).strip()
def key(t): return re.sub(r"\W+", " ", str(t).lower()).strip()

def load_csv(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    tcol = [c for c in df.columns if "text" in c.lower()][0]
    lcol = [c for c in df.columns if "type" in c.lower() or "label" in c.lower()][0]
    df = df[[tcol, lcol]].dropna(); df.columns = ["text", "label_raw"]
    df["text"] = df["text"].map(clean); df = df[df.text.str.len() > 0].reset_index(drop=True)
    df["y"] = df.label_raw.astype(str).str.lower().str.contains("phish").astype(int)
    df["key"] = df.text.map(key); return df

from sklearn.model_selection import train_test_split
def split(df, seed):
    tr, te = train_test_split(df, test_size=0.20, stratify=df.y, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.125, stratify=tr.y, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def remove_words(text, ws):
    if not ws: return text
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in ws) + r")\b", re.I)
    return re.sub(r"\s+", " ", pat.sub(" ", text)).strip()
def keep_words(text, ws):
    s = set(w.lower() for w in ws); return " ".join(w for w in WORD.findall(text) if w.lower() in s)
def minmax_pm1(d):
    v = np.array(list(d.values()), float); lo, hi = v.min(), v.max()
    if hi - lo < 1e-12: return {k: 0.0 for k in d}
    return {k: 2 * (x - lo) / (hi - lo) - 1 for k, x in d.items()}
def absmax(d):
    v = np.array(list(d.values()), float); m = np.abs(v).max()
    if m < 1e-12: return {k: 0.0 for k in d}
    return {k: x / m for k, x in d.items()}
def lita(l, t, alpha, norm):
    nl, nt = norm(l), norm(t); return {w: alpha * nl[w] + (1 - alpha) * nt[w] for w in l}
def vocab_of(text):
    seen, out = set(), []
    for w in WORD.findall(text):
        w = w.lower()
        if w not in seen: seen.add(w); out.append(w)
    return out

HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}
def atk_homoglyph(t, r=0.15, seed=0):
    rng = random.Random(seed); return "".join(HOMO[c] if c in HOMO and rng.random() < r else c for c in t)
def atk_typo(t, r=0.10, seed=0):
    rng = random.Random(seed); out = []
    for w in t.split(" "):
        if len(w) > 3 and rng.random() < r:
            i = rng.randrange(len(w) - 1); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)
def atk_zerowidth(t, r=0.10, seed=0):
    rng = random.Random(seed); out = []
    for w in t.split(" "):
        if len(w) > 3 and rng.random() < r:
            i = rng.randrange(1, len(w)); w = w[:i] + "​" + w[i:]
        out.append(w)
    return " ".join(out)
ATTACKS = {"homoglyph": atk_homoglyph, "typo": atk_typo, "zero_width": atk_zerowidth}

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(y), f1=2 * prec * rec / max(prec + rec, 1e-12),
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1), tn=tn, fp=fp, fn=fn, tp=tp)

# ---------------- data (lazy) ----------------
_D = {}
def data():
    if "raw" not in _D:
        raw = load_csv(CSV); dedup = raw.drop_duplicates("key", keep="first").reset_index(drop=True)
        _D.update(raw=raw, dedup=dedup, split=split(dedup, 42))
        print("data:", len(raw), "raw /", len(dedup), "dedup")
    return _D

# ---------------- model helpers (lazy; need torch) ----------------
M = {}
def torch_env():
    if "torch" not in M:
        import torch
        M["torch"] = torch; M["dev"] = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", M["dev"])
    return M["torch"], M["dev"]

def load_model(name):
    if name in M: return M[name]
    torch, dev = torch_env()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mdir = ckf("model", name)
    if not os.path.isdir(mdir): raise SystemExit(f"saved model not found: {mdir}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    M[name] = (model, tok); return M[name]

def predict_proba(model, tok, texts, max_len=256, bs=64):
    torch, dev = torch_env(); model.eval(); texts = list(texts); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lg = model(**enc).logits
            out.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def pick_emails(te, model, tok, n, seed):
    sub = te[te.y == 1].copy()
    sub["tx"] = sub.text.map(lambda t: " ".join(t.split()[:120]))
    sub["p"] = predict_proba(model, tok, sub.tx); sub = sub[sub.p >= .5]
    return sub.sample(min(n, len(sub)), random_state=seed).tx.tolist()

def curves(model, tok, text, scores, K=10):
    order = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])]
    texts = [text] + [remove_words(text, order[:k]) for k in range(1, K + 1)] + [keep_words(text, order[:k]) for k in range(1, K + 1)]
    p = predict_proba(model, tok, texts); return p[0], p[1:K + 1], p[K + 1:]
def faith(p0, rem, keep): return float(np.mean(p0 - rem)), float(np.mean(p0 - keep))

def exp_lime(model, tok, text, vocab, seed, n=500):
    from lime.lime_text import LimeTextExplainer
    ex = LimeTextExplainer(class_names=["safe", "phish"], random_state=seed)
    def pf(xs):
        p = predict_proba(model, tok, xs); return np.stack([1 - p, p], 1)
    e = ex.explain_instance(text, pf, num_features=len(vocab), num_samples=n, labels=(1,))
    d = {w: 0.0 for w in vocab}
    for w, s in e.as_list(label=1):
        if w.lower() in d: d[w.lower()] += s
    return d

def exp_shap(model, tok, text, vocab, max_evals):
    import shap
    f = lambda xs: predict_proba(model, tok, list(xs))
    ex = shap.Explainer(f, shap.maskers.Text(r"\W+"), algorithm="partition")
    e = ex([text], max_evals=max_evals); d = {w: 0.0 for w in vocab}
    for t, v in zip(e.data[0], e.values[0]):
        t = str(t).strip().lower()
        if t in d: d[t] += float(v if np.ndim(v) == 0 else v[-1])
    return d

def exp_ig(model, tok, text, vocab, variant):
    """variant: dict(target='prob'|'logit', steps=int, baseline='pad'|'mask', method='ig'|'gxi')"""
    torch, dev = torch_env()
    from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
    enc = tok(text, truncation=True, max_length=256, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].numpy(); enc = {k: v.to(dev) for k, v in enc.items()}
    ids = enc["input_ids"]; am = enc["attention_mask"]
    bid = tok.pad_token_id if variant.get("baseline", "pad") == "pad" else tok.mask_token_id
    base = ids.clone(); base[:, 1:-1] = bid
    emb = model.base_model.embeddings
    if variant.get("target", "prob") == "prob":
        fwd = lambda x, m: torch.softmax(model(input_ids=x, attention_mask=m).logits.float(), -1)[:, 1]
    else:
        def fwd(x, m):
            lg = model(input_ids=x, attention_mask=m).logits.float(); return lg[:, 1] - lg[:, 0]
    delta = float("nan")
    if variant.get("method", "ig") == "ig":
        lig = LayerIntegratedGradients(fwd, emb)
        at, dl = lig.attribute(ids, baselines=base, additional_forward_args=(am,), n_steps=variant["steps"],
                               internal_batch_size=16, return_convergence_delta=True)
        with torch.no_grad(): gap = float((fwd(ids, am) - fwd(base, am)).abs())
        delta = float(dl.abs().mean()) / max(gap, 1e-9)       # relative completeness error
    else:
        gx = LayerGradientXActivation(fwd, emb); at = gx.attribute(ids, additional_forward_args=(am,))
    sc = at.sum(-1)[0].detach().cpu().numpy()
    spans = [(m.start(), m.end(), m.group().lower()) for m in WORD.finditer(text)]
    d = {w: 0.0 for w in vocab}
    for (s0, e0), s in zip(offs, sc):
        if e0 <= s0: continue
        for ws, we, w in spans:
            if s0 < we and e0 > ws:
                if w in d: d[w] += float(s)
                break
    return d, delta

# ---------------- shared helpers for the classical models ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

def tfidf(ng=(1, 2)): return TfidfVectorizer(max_features=50000, ngram_range=ng, sublinear_tf=True, min_df=2)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def mrow(y, p, **extra):
    """accuracy, precision, recall, F1, FPR, FNR (+ Wilson 95% CI of the accuracy)"""
    y = np.asarray(y); p = np.asarray(p); m = metrics(y, p)
    prec = m["tp"] / max(m["tp"] + m["fp"], 1); rec = m["tp"] / max(m["tp"] + m["fn"], 1)
    lo, hi = wilson(int((y == p).sum()), len(y))
    return dict(**extra, n=len(y), acc=m["acc"], acc_lo=lo, acc_hi=hi, prec=prec, rec=rec, f1=m["f1"], fpr=m["fpr"], fnr=m["fnr"])

def fit_default(tr, te):
    """the paper's fixed settings: LR C=10, SVM C=1, RF 300 trees; all class-balanced; word 1-2-grams"""
    vec = tfidf(); Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text); out = {}
    out["LR"] = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, tr.y).predict(Xte)
    out["SVM"] = LinearSVC(C=1.0, class_weight="balanced").fit(Xtr, tr.y).predict(Xte)
    if a.with_rf:
        out["RF"] = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0).fit(Xtr, tr.y).predict(Xte)
    return out

GRID_LR = [0.1, 0.3, 1, 3, 10, 30, 100]; GRID_SVM = [0.03, 0.1, 0.3, 1, 3, 10]; GRID_NG = [(1, 1), (1, 2)]
def fit_tuned(tr, va, te):
    """grid search on the VALIDATION split only (macro-selection by F1); the test split is touched once, after selection"""
    best = {"LR": (-1, None, None), "SVM": (-1, None, None)}
    for ng in GRID_NG:
        vec = tfidf(ng); Xtr = vec.fit_transform(tr.text); Xva = vec.transform(va.text); Xte = vec.transform(te.text)
        for C in GRID_LR:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, tr.y)
            f = metrics(va.y.values, clf.predict(Xva))["f1"]
            if f > best["LR"][0]: best["LR"] = (f, dict(ngram=str(ng), C=C), clf.predict(Xte))
        for C in GRID_SVM:
            clf = LinearSVC(C=C, class_weight="balanced").fit(Xtr, tr.y)
            f = metrics(va.y.values, clf.predict(Xva))["f1"]
            if f > best["SVM"][0]: best["SVM"] = (f, dict(ngram=str(ng), C=C), clf.predict(Xte))
    return best

def summarize(df, keys, cols=("acc", "f1", "fpr", "fnr")):
    g = df.groupby(keys)[list(cols)].agg(["mean", "std"]) * 100
    print(g.round(2).to_string())

def load_probs(proto, seed, name):
    f = ckf("train", proto, seed, name) + ".pkl"
    return pickle.load(open(f, "rb"))["prob"] if os.path.exists(f) else None

def paired_boot(ca, cb, n=5000, seed=0):
    rng = np.random.RandomState(seed); N = len(ca); d = (ca - cb)
    bs = [d[rng.randint(0, N, N)].mean() * 100 for _ in range(n)]
    return d.mean() * 100, *np.percentile(bs, [2.5, 97.5])

def mcnemar_exact(ca, cb):
    b = int(((ca == 1) & (cb == 0)).sum()); c = int(((ca == 0) & (cb == 1)).sum())
    return stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0

# =====================================================================================
if "tuned" in STEPS:
    print("\n=== tuned: validation-tuned TF-IDF baselines (grid: n-gram x C), dedup, seeds", SEEDS, "===")
    D = data(); rows = []
    for seed in SEEDS:
        tr, va, te = split(D["dedup"], seed); best = fit_tuned(tr, va, te)
        for mname, (vf1, cfg, pred) in best.items():
            rows.append(mrow(te.y.values, pred, model=mname + "_tuned", seed=seed, val_f1=vf1, **{f"cfg_{k}": v for k, v in cfg.items()}))
        for mname, pred in fit_default(tr, te).items():
            rows.append(mrow(te.y.values, pred, model=mname + "_default", seed=seed))
        print("seed", seed, "done", flush=True)
    R = pd.DataFrame(rows); save(R, "tuned_baselines.csv"); summarize(R, "model")
    print(R[R.model.str.endswith("_tuned")][["model", "seed", "val_f1", "cfg_ngram", "cfg_C", "acc"]].round(4).to_string())

# =====================================================================================
if "matched" in STEPS:
    print("\n=== matched: TF-IDF on the same first-256-token text the transformers see ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    def trunc(texts, L=256):
        enc = tok(list(texts), truncation=True, max_length=L, return_offsets_mapping=True)
        out = []
        for t, offs in zip(texts, enc["offset_mapping"]):
            ends = [e for s, e in offs if e > s]; out.append(t[:ends[-1]] if ends else t)
        return out
    D = data(); rows, pairs = [], []
    for seed in SEEDS:
        tr, va, te = split(D["dedup"], seed)
        tr2, va2, te2 = [d.assign(text=trunc(d.text.tolist())) for d in (tr, va, te)]
        print("seed", seed, "mean words kept:", round(np.mean([len(t.split()) for t in te2.text]), 1),
              "of", round(np.mean([len(t.split()) for t in te.text]), 1), flush=True)
        best = fit_tuned(tr2, va2, te2); preds = {}
        for mname, (vf1, cfg, pred) in best.items():
            rows.append(mrow(te2.y.values, pred, model=mname + "_tuned_trunc256", seed=seed, val_f1=vf1, **{f"cfg_{k}": v for k, v in cfg.items()})); preds[mname + "_tuned_trunc256"] = pred
        for mname, pred in fit_default(tr2, te2).items():
            rows.append(mrow(te2.y.values, pred, model=mname + "_default_trunc256", seed=seed)); preds[mname + "_default_trunc256"] = pred
        y = te2.y.values
        for tname in MODELS:
            pr = load_probs("dedup", seed, tname)
            if pr is None: continue
            ct = ((pr >= .5).astype(int) == y).astype(float)
            for bname, pred in preds.items():
                cb = (pred == y).astype(float); d, lo, hi = paired_boot(ct, cb)
                pairs.append(dict(seed=seed, transformer=tname, baseline=bname, diff_pts=d, ci_lo=lo, ci_hi=hi, mcnemar_p=mcnemar_exact(ct, cb)))
    R = pd.DataFrame(rows); save(R, "matched_input.csv"); summarize(R, "model")
    if pairs: P = pd.DataFrame(pairs); save(P, "matched_paired.csv"); print(P.round(3).to_string())

# =====================================================================================
def sim_groups(texts, thr):
    """connected components of the graph 'cosine(TF-IDF 1-2-gram) >= thr' (union-find)"""
    X = tfidf().fit_transform(texts); n = X.shape[0]; XT = X.T.tocsr(); parent = np.arange(n)
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i0 in range(0, n, 400):
        S = (X[i0:i0 + 400] @ XT).tocoo(); m = (S.data >= thr) & (S.row + i0 < S.col)
        for r, c in zip(S.row[m] + i0, S.col[m]):
            ra, rb = find(r), find(c)
            if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])

def group_split(df, groups, seed):
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_i, te_i = next(iter(sg.split(df, df.y, groups)))
    rest = df.iloc[tr_i]; g2 = groups[tr_i]
    sg2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    t2, v2 = next(iter(sg2.split(rest, rest.y, g2)))
    return rest.iloc[t2].reset_index(drop=True), rest.iloc[v2].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)

def train_transformer(name, tr, va, te, seed, tag):
    """same recipe as the main notebook (AdamW 2e-5, wd .01, batch 32, 3 epochs, 10% warm-up, fp16, class-weighted CE,
    clip 1.0, 256 tokens, best validation F1 epoch); validation predictions are made in eval mode"""
    cf = ckf("round2", tag, seed, name) + ".pkl"
    if os.path.exists(cf): print("resumed", tag, seed, name); return pickle.load(open(cf, "rb"))["prob"]
    torch, dev = torch_env()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    n_steps = 3 * math.ceil(len(tr) / 32)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * n_steps), max(0.0, (n_steps - s) / (0.9 * n_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    freq = tr.y.value_counts(normalize=True).sort_index().values
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(1.0 / (2 * freq), dtype=torch.float, device=dev))
    best, best_state = -1, None; X, Y = tr.text.tolist(), tr.y.values
    for ep in range(3):
        model.train(); idx = np.random.RandomState(seed + ep).permutation(len(X))
        for i in range(0, len(X), 32):
            b = idx[i:i + 32]
            enc = tok([X[j] for j in b], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lg = model(**enc).logits
            loss = lossf(lg.float(), torch.tensor(Y[b], device=dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        pv = predict_proba(model, tok, va.text); f1 = metrics(va.y.values, (pv >= .5).astype(int))["f1"]
        print(tag, name, "seed", seed, "epoch", ep + 1, "val f1", round(f1, 4), flush=True)
        if f1 > best: best = f1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval(); prob = predict_proba(model, tok, te.text)
    pickle.dump(dict(prob=prob), open(cf, "wb")); del model; torch.cuda.empty_cache(); return prob

if "neardup" in STEPS:
    print("\n=== neardup: near-duplicate leakage in the dedup split, and group-aware splits ===")
    D = data(); dd = D["dedup"]; THR = (0.95, 0.90)
    # (1) how many dedup-split test emails have a near-duplicate in training, and is accuracy different on them?
    tr, va, te = D["split"]; vec = tfidf(); Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text); XtrT = Xtr.T.tocsr()
    maxsim = np.concatenate([np.asarray((Xte[i:i + 400] @ XtrT).max(axis=1).todense()).ravel() for i in range(0, Xte.shape[0], 400)])
    base = fit_default(tr, te); y = te.y.values; rows = []
    cor = {k: (v == y) for k, v in base.items()}
    for tname in MODELS:
        pr = load_probs("dedup", 42, tname)
        if pr is not None: cor[tname] = ((pr >= .5).astype(int) == y)
    for thr in THR:
        fl = maxsim >= thr
        for mname, c in cor.items():
            rows.append(dict(thr=thr, model=mname, n_flagged=int(fl.sum()), frac_flagged=float(fl.mean()),
                             acc_flagged=float(c[fl].mean()) if fl.any() else np.nan, acc_unflagged=float(c[~fl].mean())))
    L = pd.DataFrame(rows); save(L, "neardup_leakage_dedup_seed42.csv"); print(L.round(4).to_string())
    # (2) group-aware splits: no near-duplicate cluster crosses train/val/test
    grows, stat_rows = [], []
    for thr in THR:
        g = sim_groups(dd.text, thr); sizes = pd.Series(g).value_counts()
        stat_rows.append(dict(thr=thr, n=len(dd), n_groups=int(len(sizes)), largest_group=int(sizes.iloc[0]),
                              frac_in_multi=float((sizes[sizes > 1]).sum() / len(dd))))
        print(stat_rows[-1], flush=True)
        for seed in SEEDS:
            tr, va, te = group_split(dd, g, seed)
            print(f" thr {thr} seed {seed}: train/val/test = {len(tr)}/{len(va)}/{len(te)}, test phishing share {te.y.mean():.3f}", flush=True)
            for mname, pred in fit_default(tr, te).items():
                grows.append(mrow(te.y.values, pred, thr=thr, seed=seed, model=mname))
            best = fit_tuned(tr, va, te)
            for mname, (vf1, cfg, pred) in best.items():
                grows.append(mrow(te.y.values, pred, thr=thr, seed=seed, model=mname + "_tuned", **{f"cfg_{k}": v for k, v in cfg.items()}))
            if a.nd_transformers and abs(thr - a.nd_tf_thr) < 1e-9 and seed in [int(s) for s in a.nd_seeds.split(",")]:
                for tname in a.nd_transformers.split(","):
                    pr = train_transformer(tname, tr, va, te, seed, f"nd{thr}")
                    grows.append(mrow(te.y.values, (pr >= .5).astype(int), thr=thr, seed=seed, model=tname))
                save(pd.DataFrame(grows), "neardup_group_split.csv")      # partial save after every transformer
    save(pd.DataFrame(stat_rows), "neardup_group_stats.csv"); G = pd.DataFrame(grows); save(G, "neardup_group_split.csv")
    summarize(G, ["thr", "model"])

# =====================================================================================
if "empty" in STEPS:
    print("\n=== empty: sensitivity to the conflicting 'empty' group ===")
    D = data(); raw = D["raw"]; rows = []
    em = raw[raw.key == "empty"]; maj = int(em.y.mean() >= .5)
    print("'empty' rows:", len(em), "phishing:", int(em.y.sum()), "majority label:", maj)
    def build(corpus, policy):
        if policy == "first_copy (paper)": return corpus.drop_duplicates("key", keep="first").reset_index(drop=True)
        base = corpus[corpus.key != "empty"].drop_duplicates("key", keep="first")
        if policy == "group_removed": return base.reset_index(drop=True)
        if policy == "majority_label_rep":
            rep = pd.DataFrame([dict(text="empty", label_raw="", y=maj, key="empty")])
            return pd.concat([base, rep], ignore_index=True)
    for policy in ("first_copy (paper)", "group_removed", "majority_label_rep"):
        dd = build(raw, policy)
        for seed in SEEDS:
            tr, va, te = split(dd, seed)
            for mname, pred in fit_default(tr, te).items():
                rows.append(mrow(te.y.values, pred, protocol="dedup", policy=policy, seed=seed, model=mname, n_empty_test=int((te.key == "empty").sum())))
        print(policy, "corpus size", len(dd), flush=True)
    # raw (non-deduplicated) split, seed 42: whole group kept (paper) vs removed from the corpus
    for policy, corpus in (("group_kept (paper)", raw), ("group_removed", raw[raw.key != "empty"].reset_index(drop=True))):
        tr, va, te = split(corpus, 42)
        for mname, pred in fit_default(tr, te).items():
            rows.append(mrow(te.y.values, pred, protocol="raw", policy=policy, seed=42, model=mname, n_empty_test=int((te.key == "empty").sum())))
    # transformers: saved test probabilities, dedup test set with the 'empty' rows dropped
    tr, va, te = D["split"]
    for seed in (42, 43, 44):
        trs, vas, tes = split(D["dedup"], seed)
        for tname in MODELS:
            pr = load_probs("dedup", seed, tname)
            if pr is None: continue
            keep = (tes.key != "empty").values
            rows.append(mrow(tes.y.values[keep], (pr[keep] >= .5).astype(int), protocol="dedup", policy="empty_rows_dropped_from_test", seed=seed,
                             model=tname, n_empty_test=int((~keep).sum())))
            rows.append(mrow(tes.y.values, (pr >= .5).astype(int), protocol="dedup", policy="first_copy (paper)", seed=seed, model=tname,
                             n_empty_test=int((~keep).sum())))
    R = pd.DataFrame(rows); save(R, "empty_sensitivity.csv"); summarize(R, ["protocol", "policy", "model"])

# =====================================================================================
if "enron_lr" in STEPS:
    print("\n=== enron_lr: classical models on the external samples, incl. non-overlapping subset ===")
    from datasets import load_dataset
    D = data(); raw_keys = set(D["raw"].key); tr, va, te = D["split"]; rows = []
    vec = tfidf(); Xtr = vec.fit_transform(tr.text)
    clfs = {"LR": LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, tr.y),
            "SVM": LinearSVC(C=1.0, class_weight="balanced").fit(Xtr, tr.y)}
    for dsid in ["SetFit/enron_spam", "zefang-liu/phishing-email-dataset"]:
        try:
            d = load_dataset(dsid, split="train").to_pandas()
            tcol = [c for c in d.columns if c.lower() in ("text", "email text", "email", "body", "message")][0]
            lcol = [c for c in d.columns if "label" in c.lower() or "type" in c.lower()][0]
            yy = d[lcol]
            if yy.dtype == object: yy = yy.astype(str).str.lower().str.contains("phish|spam").astype(int)
            dfx = pd.DataFrame(dict(text=d[tcol].map(clean), y=yy.astype(int))).dropna()
            dfx = dfx[dfx.text.str.len() > 0].sample(min(3000, len(dfx)), random_state=0).reset_index(drop=True)
            dfx["key"] = dfx.text.map(key); inraw = dfx.key.isin(raw_keys).values; Xe = vec.transform(dfx.text)
            for mname, clf in clfs.items():
                pr = clf.predict(Xe)
                rows.append(dict(dataset=dsid, model=mname, n=len(dfx), acc_all=float((pr == dfx.y).mean()), n_nonoverlap=int((~inraw).sum()),
                                 acc_nonoverlap=float((pr[~inraw] == dfx.y.values[~inraw]).mean()) if (~inraw).any() else np.nan))
        except Exception as e:
            rows.append(dict(dataset=dsid, error=str(e)[:200]))
    save(pd.DataFrame(rows), "external_classical.csv"); print(pd.DataFrame(rows).round(4).to_string())

# =====================================================================================
# GPU steps: explanations
IGPAD = dict(target="prob", steps=30, baseline="pad", method="ig")                 # the paper's default IG
IGMASK = dict(target="logit", steps=100, baseline="mask", method="ig")             # the configuration that matched LIME

if "lime_repeat" in STEPS:
    print("\n=== lime_repeat: same checkpoint, same emails, LIME re-run with new random seeds ===")
    D = data(); tr, va, te = D["split"]; R = a.n_lime_repeats; rows = []
    for name in MODELS:
        model, tok = load_model(name); ts = pick_emails(te, model, tok, a.n_expl, 1)
        cf = os.path.join(OUT, f"lime_repeat_cache_{name}.pkl"); cache = pickle.load(open(cf, "rb")) if os.path.exists(cf) else {}
        t0 = time.time()
        for i, tx in enumerate(ts):
            if i in cache: continue
            vocab = vocab_of(tx); igp = exp_ig(model, tok, tx, vocab, IGPAD)[0]; igm = exp_ig(model, tok, tx, vocab, IGMASK)[0]
            igp2 = exp_ig(model, tok, tx, vocab, IGPAD)[0]                          # determinism check for IG
            ig_diff = float(max(abs(igp[w] - igp2[w]) for w in vocab))
            rec = {}; L0 = None
            for r in range(R + 1):
                seed = 100 + i if r == 0 else 100000 * r + i                         # r = 0 reproduces the paper's LIME run
                L = exp_lime(model, tok, tx, vocab, seed); L0 = L if r == 0 else L0
                lc, ls = faith(*curves(model, tok, tx, L))
                hm = faith(*curves(model, tok, tx, lita(L, igm, .5, absmax)))[0]
                hp = faith(*curves(model, tok, tx, lita(L, igp, .5, absmax)))[0]
                rho = float(stats.spearmanr([L0[w] for w in vocab], [L[w] for w in vocab])[0]) if r else 1.0
                rec[r] = dict(lime_comp=lc, lime_suff=ls, hyb_mask_comp=hm, hyb_pad_comp=hp, rho_vs_run0=rho)
            cache[i] = dict(rec=rec, ig_maxdiff=ig_diff)
            if (i + 1) % 5 == 0 or i + 1 == len(ts):
                pickle.dump(cache, open(cf, "wb")); print(f"{name}: {i+1}/{len(ts)} emails, {(time.time()-t0)/60:.1f} min", flush=True)
        for i, c in cache.items():
            for r, v in c["rec"].items(): rows.append(dict(model=name, email=i, repeat=r, ig_maxdiff=c["ig_maxdiff"], **v))
    P = pd.DataFrame(rows); save(P, "lime_repeat_per_email.csv")
    S = P.groupby(["model", "repeat"]).agg(n=("email", "count"), lime_comp=("lime_comp", "mean"), hyb_mask_comp=("hyb_mask_comp", "mean"),
                                           hyb_pad_comp=("hyb_pad_comp", "mean"), rho_vs_run0=("rho_vs_run0", "mean")).reset_index()
    S["hyb_mask_minus_lime"] = S.hyb_mask_comp - S.lime_comp; S["hyb_pad_minus_lime"] = S.hyb_pad_comp - S.lime_comp
    save(S, "lime_repeat_summary.csv"); print(S.round(4).to_string())
    print("max |IG(pad) run1 - run2| over all emails:", P.groupby("model").ig_maxdiff.max().to_dict())

# =====================================================================================
if "strat" in STEPS:
    print("\n=== strat: the same stratified emails explained for BOTH models (phishing + safe, confident + borderline + errors) ===")
    D = data(); tr, va, te = D["split"]; rng = np.random.RandomState(7)
    models = {n: load_model(n) for n in MODELS}
    te = te.copy(); te["tx"] = te.text.map(lambda t: " ".join(t.split()[:120]))
    te = te[te.tx.map(lambda t: len(vocab_of(t))) >= 15].reset_index(drop=True)
    pc, pred = {}, {}
    for n, (m, tk) in models.items():
        p = predict_proba(m, tk, te.tx); pred[n] = (p >= .5).astype(int); pc[n] = np.where(pred[n] == 1, p, 1 - p)
    ok = np.ones(len(te), bool)
    for n in MODELS: ok &= (pred[n] == te.y.values)
    conf = np.minimum(*[pc[n] for n in MODELS]); y = te.y.values; half = a.n_strat // 2
    strata = {"phishing_ok_confident": (y == 1) & ok & (conf >= .99), "phishing_ok_borderline": (y == 1) & ok & (conf < .99),
              "safe_ok_confident": (y == 0) & ok & (conf >= .99), "safe_ok_borderline": (y == 0) & ok & (conf < .99), "model_error": ~ok}
    chosen = {}
    for sname, mask in strata.items():
        idx = np.where(mask)[0]; k = a.n_strat if sname == "model_error" else half
        chosen[sname] = sorted(rng.choice(idx, min(k, len(idx)), replace=False).tolist())
        print(f"  {sname}: available {len(idx)}, using {len(chosen[sname])}")
    sample = [(s, i) for s, ids in chosen.items() for i in ids]
    json.dump(dict(strata=chosen), open(os.path.join(OUT, "strat_sample.json"), "w"))
    methods = ["LIME", "SHAP", "IG_pad", "IG_mask", "Hybrid_pad", "Hybrid_mask", "Random"]
    rows = []
    for n, (m, tk) in models.items():
        cf = os.path.join(OUT, f"strat_cache_{n}.pkl"); cache = pickle.load(open(cf, "rb")) if os.path.exists(cf) else {}
        t0 = time.time()
        for j, (sname, i) in enumerate(sample):
            if i in cache: continue
            tx = te.tx[i]; vocab = vocab_of(tx); sign = 1 if pred[n][i] == 1 else -1
            L = exp_lime(m, tk, tx, vocab, 100 + i); S_ = exp_shap(m, tk, tx, vocab, 300)
            Tp = exp_ig(m, tk, tx, vocab, IGPAD)[0]; Tm = exp_ig(m, tk, tx, vocab, IGMASK)[0]
            rg = np.random.RandomState(1000 + i); Rn = {w: rg.randn() for w in vocab}
            sc = {"LIME": L, "SHAP": S_, "IG_pad": Tp, "IG_mask": Tm, "Hybrid_pad": lita(L, Tp, .5, absmax),
                  "Hybrid_mask": lita(L, Tm, .5, absmax), "Random": Rn}
            rec = {}
            for mn, s in sc.items():
                if sign < 0: s = {w: -v for w, v in s.items()}              # rank the most SAFE-indicative words first
                p0, rem, keep = curves(m, tk, tx, s)
                if sign < 0: p0, rem, keep = 1 - p0, 1 - rem, 1 - keep       # work with the probability of the predicted class
                c, su = faith(p0, rem, keep); rec[mn] = (c, su)
            cache[i] = dict(rec=rec, p0=float(pc[n][i]))
            if (j + 1) % 5 == 0 or j + 1 == len(sample):
                pickle.dump(cache, open(cf, "wb")); print(f"{n}: {j+1}/{len(sample)} emails, {(time.time()-t0)/60:.1f} min", flush=True)
        pickle.dump(cache, open(cf, "wb"))
        for sname, i in sample:
            for mn, (c, su) in cache[i]["rec"].items():
                rows.append(dict(model=n, email=i, stratum=sname, y=int(y[i]), pred=int(pred[n][i]), p_pred=cache[i]["p0"], method=mn, comp=c, suff=su))
    P = pd.DataFrame(rows); save(P, "strat_faithfulness.csv")
    print(P.pivot_table(index=["model", "stratum"], columns="method", values="comp", aggfunc="mean").round(4).to_string())
    print(P.pivot_table(index="model", columns="method", values="comp", aggfunc="mean").round(4).to_string())

# =====================================================================================
if "reseeds" in STEPS:
    print("\n=== reseeds: RoBERTa-base seeds 43, 44 retrained with eval-mode validation (matches the main notebook exactly) ===")
    D = data(); rows = []
    for seed in (43, 44):
        tr, va, te = split(D["dedup"], seed); pr = train_transformer("roberta-base", tr, va, te, seed, "reseed")
        rows.append(mrow(te.y.values, (pr >= .5).astype(int), seed=seed, model="roberta-base"))
    save(pd.DataFrame(rows), "roberta_reseeds.csv"); print(pd.DataFrame(rows).round(4).to_string())

# ---------------- zip ----------------
zp = os.path.join(FULL, "round2_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".csv", ".json")): z.write(os.path.join(OUT, f), f)
print("\nDone. Send me:", zp)
try:
    from google.colab import files; files.download(zp)
except Exception: pass
