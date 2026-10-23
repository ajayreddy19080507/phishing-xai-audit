# =====================================================================================
# controls_colab.py  --  control experiments for the phishing-XAI paper (Colab T4)
#
# Run AFTER the main notebook has finished (it reads its Drive outputs and saved models).
# In Colab:   (1) Runtime > T4 GPU   (2) upload this file   (3) a cell with:
#       !pip -q install transformers captum lime shap scikit-learn scipy datasets
#       !python controls_colab.py --drive /content/drive/MyDrive/phish_xai --csv /content/drive/MyDrive/phish_xai/Phishing_Email.csv
#   (mount Drive first:  from google.colab import drive; drive.mount('/content/drive'))
#
# Every step checkpoints into  <drive>/full/controls/  and is skipped on re-run.
# Steps (select with --steps, default all but 'seeds'):
#   export   per-email faithfulness table + mean curves from the saved checkpoints   (CPU, seconds)
#   paired   paired-bootstrap accuracy differences + equivalence (TOST) check        (CPU, ~2 min)
#   overlap  exact-text overlap of Enron-Spam / zefang-liu samples with the Kaggle corpus
#   robust   token-length / truncation controls for the homoglyph / zero-width / typo tests
#   ig       IG controls: logit target, more steps, mask baseline, Gradient x Input   (GPU, ~30 min)
#   shap     SHAP budget sensitivity (300 vs 1000 vs 2000 evaluations, 30 emails)    (GPU, ~10 min)
#   seeds    OPTIONAL: train RoBERTa-base for seeds 43 and 44 (about 30-40 min each)
# Output: <drive>/full/controls/*.csv and controls_results.zip  -> send the zip back.
# =====================================================================================
import argparse, ast, json, math, os, pickle, random, re, sys, time, zipfile
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--drive", default="/content/drive/MyDrive/phish_xai")
ap.add_argument("--csv", default=None)
ap.add_argument("--steps", default="export,paired,overlap,robust,ig,shap")
ap.add_argument("--n", type=int, default=100, help="emails per model for the IG controls (default 100 = same as the paper)")
ap.add_argument("--n-shap", type=int, default=30)
ap.add_argument("--models", default="distilroberta-base,roberta-base")
a = ap.parse_args()
STEPS = set(a.steps.split(","))
MODELS = a.models.split(",")
FULL = os.path.join(a.drive, "full"); CKPT = os.path.join(FULL, "ckpt"); RES = os.path.join(FULL, "results")
OUT = os.path.join(FULL, "controls"); os.makedirs(OUT, exist_ok=True)
CSV = a.csv or os.path.join(a.drive, "Phishing_Email.csv")
def ckf(*p): return os.path.join(CKPT, "_".join(str(x) for x in p))
def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False); print(f"[saved] {name}  ({len(df)} rows)")

# ---------------- helpers copied from the main notebook (same definitions) ----------------
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
    torch, dev = torch_env(); texts = list(texts); out = []
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

# =====================================================================================
if "export" in STEPS:
    print("\n=== export: per-email faithfulness + mean curves ===")
    rows, curv = [], {}
    for name in MODELS:
        f = ckf("expl", name) + ".pkl"
        if not os.path.exists(f): print("missing", f); continue
        recs = pickle.load(open(f, "rb"))
        for i, r in enumerate(recs):
            for m, v in r["res"].items():
                rows.append(dict(model=name, email=i, method=m, comp=v[0], suff=v[1]))
        for m in recs[0]["res"]:
            rem = np.array([r["res"][m][2] for r in recs]); keep = np.array([r["res"][m][3] for r in recs])
            curv[f"{name}|{m}"] = dict(rem_mean=rem.mean(0).tolist(), rem_sd=rem.std(0).tolist(),
                                       keep_mean=keep.mean(0).tolist(), keep_sd=keep.std(0).tolist(), n=len(recs))
    save(pd.DataFrame(rows), "per_email_faithfulness.csv")
    json.dump(curv, open(os.path.join(OUT, "curves_export.json"), "w"))
    # alpha-sweep bootstrap needs the val cache
    for name in MODELS:
        vf = ckf("valcache", name) + ".pkl"
        if os.path.exists(vf):
            vc = pickle.load(open(vf, "rb")); print(name, "val cache emails:", len(vc))

# =====================================================================================
if "paired" in STEPS:
    print("\n=== paired: paired-bootstrap accuracy differences and equivalence (+-0.5 pt) ===")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    tr, va, te = data()["split"]; y = te.y.values
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text)
    P = {"LR": LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, tr.y).predict(Xte),
         "SVM": LinearSVC(C=1.0, class_weight="balanced").fit(Xtr, tr.y).predict(Xte)}
    for name in MODELS:
        f = ckf("train", "dedup", 42, name) + ".pkl"
        if os.path.exists(f): P[name] = (pickle.load(open(f, "rb"))["prob"] >= .5).astype(int)
    rng = np.random.RandomState(0); n = len(y); idx = [rng.randint(0, n, n) for _ in range(5000)]
    rows = []
    for tname in [m for m in MODELS if m in P]:
        for bname in ("LR", "SVM"):
            ca = (P[tname] == y).astype(float); cb = (P[bname] == y).astype(float)
            d = [(ca[i] - cb[i]).mean() * 100 for i in idx]; lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(a=tname, b=bname, diff_pts=(ca - cb).mean() * 100, ci_lo=lo, ci_hi=hi,
                             equivalent_within_0p5=bool(lo > -0.5 and hi < 0.5), n=n))
    # also the 90% CI used by TOST
    save(pd.DataFrame(rows), "paired_accuracy_diff.csv"); print(pd.DataFrame(rows).round(3).to_string())

# =====================================================================================
if "overlap" in STEPS:
    print("\n=== overlap: external samples vs the Kaggle corpus (exact normalised text) ===")
    from datasets import load_dataset
    D = data(); raw_keys = set(D["raw"].key); tr_keys = set(D["split"][0].key)
    rows = []
    for dsid in ["SetFit/enron_spam", "zefang-liu/phishing-email-dataset"]:
        try:
            d = load_dataset(dsid, split="train").to_pandas()
            tcol = [c for c in d.columns if c.lower() in ("text", "email text", "email", "body", "message")][0]
            lcol = [c for c in d.columns if "label" in c.lower() or "type" in c.lower()][0]
            yy = d[lcol]
            if yy.dtype == object: yy = yy.astype(str).str.lower().str.contains("phish|spam").astype(int)
            dfx = pd.DataFrame(dict(text=d[tcol].map(clean), y=yy.astype(int))).dropna()
            dfx = dfx[dfx.text.str.len() > 0].sample(min(3000, len(dfx)), random_state=0).reset_index(drop=True)
            dfx["key"] = dfx.text.map(key)
            in_raw = dfx.key.isin(raw_keys).values; in_tr = dfx.key.isin(tr_keys).values
            r = dict(dataset=dsid, n=len(dfx), in_kaggle_any=int(in_raw.sum()), frac_in_kaggle=float(in_raw.mean()),
                     in_train_seed42=int(in_tr.sum()))
            for name in MODELS:
                model, tok = load_model(name); pr = (predict_proba(model, tok, dfx.text) >= .5).astype(int)
                r[f"acc_all_{name}"] = float((pr == dfx.y).mean())
                if (~in_raw).sum(): r[f"acc_nonoverlap_{name}"] = float((pr[~in_raw] == dfx.y.values[~in_raw]).mean())
                r[f"n_nonoverlap"] = int((~in_raw).sum())
            rows.append(r)
        except Exception as e:
            rows.append(dict(dataset=dsid, error=str(e)[:200]))
    save(pd.DataFrame(rows), "external_overlap.csv"); print(pd.DataFrame(rows).T.to_string())

# =====================================================================================
if "robust" in STEPS:
    print("\n=== robust: truncation / token-length controls for the perturbation tests ===")
    tr, va, te = data()["split"]; y = te.y.values; rows = []
    for name in MODELS:
        model, tok = load_model(name)
        clean_len = np.array([len(x) for x in tok(te.text.tolist(), add_special_tokens=True)["input_ids"]])
        p0 = predict_proba(model, tok, te.text) >= .5
        rows.append(dict(model=name, attack="none", max_len=256, mean_tokens=clean_len.mean(), frac_over_256=(clean_len > 256).mean(), **metrics(y, p0.astype(int))))
        for an, fn in ATTACKS.items():
            pt = [fn(t, seed=i) for i, t in enumerate(te.text)]
            plen = np.array([len(x) for x in tok(pt, add_special_tokens=True)["input_ids"]])
            for ml in (256, 512):
                pr = predict_proba(model, tok, pt, max_len=ml) >= .5
                rows.append(dict(model=name, attack=an, max_len=ml, mean_tokens=plen.mean(), frac_over_256=(plen > 256).mean(), **metrics(y, pr.astype(int))))
            keepm = plen <= 256                     # emails whose perturbed version still fits in 256 tokens
            pr = predict_proba(model, tok, [t for t, k in zip(pt, keepm) if k], max_len=256) >= .5
            pc_ = p0[keepm]
            rows.append(dict(model=name, attack=an + "_fits256_only", max_len=256, mean_tokens=plen[keepm].mean(), frac_over_256=0.0,
                             n=int(keepm.sum()), **metrics(y[keepm], pr.astype(int)),
                             fpr_clean_same_subset=metrics(y[keepm], pc_.astype(int))["fpr"]))
    save(pd.DataFrame(rows), "robust_controls.csv")
    print(pd.DataFrame(rows)[["model", "attack", "max_len", "mean_tokens", "frac_over_256", "acc", "f1", "fpr", "fnr"]].round(4).to_string())
    # Cyrillic prevalence by class in the training corpus
    raw = data()["raw"]; cy = raw.text.map(lambda t: bool(re.search(r"[Ѐ-ӿ]", t)))
    ct = pd.crosstab(raw.y.map({0: "safe", 1: "phish"}), cy); print(ct); ct.to_csv(os.path.join(OUT, "cyrillic_by_class.csv"))

# =====================================================================================
if "ig" in STEPS or "shap" in STEPS:
    VARIANTS = {   # name -> settings
        "IG_prob30 (paper)":      dict(target="prob", steps=30, baseline="pad", method="ig"),
        "IG_prob300":             dict(target="prob", steps=300, baseline="pad", method="ig"),
        "IG_logit100":            dict(target="logit", steps=100, baseline="pad", method="ig"),
        "IG_logit300":            dict(target="logit", steps=300, baseline="pad", method="ig"),
        "IG_logit100_maskbase":   dict(target="logit", steps=100, baseline="mask", method="ig"),
        "GradxInput_logit":       dict(target="logit", steps=0, baseline="pad", method="gxi"),
    }

if "ig" in STEPS:
    print("\n=== ig: IG controls (target, steps, baseline) ===")
    D = data(); tr, va, te = D["split"]; res_rows, per_rows = [], []
    for name in MODELS:
        model, tok = load_model(name)
        ts = pick_emails(te, model, tok, a.n, 1)
        if name == "distilroberta-base":                    # sanity: same emails as the paper
            try:
                ex = json.load(open(os.path.join(RES, "example_attributions.json")))
                print("same first email as paper:", ts[0] == ex["text"])
            except Exception as e: print("could not compare first email:", e)
        cf = os.path.join(OUT, f"ig_cache_{name}.pkl")
        cache = pickle.load(open(cf, "rb")) if os.path.exists(cf) else {}
        t0 = time.time()
        for i, tx in enumerate(ts):
            if i in cache: continue
            vocab = vocab_of(tx); L = exp_lime(model, tok, tx, vocab, 100 + i)
            rng = np.random.RandomState(1000 + i); Rn = {w: rng.randn() for w in vocab}
            rec = {"LIME": (L, np.nan), "Random": (Rn, np.nan)}
            for vn, vs in VARIANTS.items(): rec[vn] = exp_ig(model, tok, tx, vocab, vs)
            out = {}
            for vn, (sc, dl) in rec.items():
                c, s = faith(*curves(model, tok, tx, sc)); out[vn] = dict(comp=c, suff=s, delta=dl)
                if vn.startswith(("IG", "Grad")):
                    hc, hs = faith(*curves(model, tok, tx, lita(L, sc, .5, absmax)))
                    out[vn].update(h_comp=hc, h_suff=hs, spearman_lime=float(stats.spearmanr(list(L.values()), list(sc.values()))[0]))
            cache[i] = out
            if (i + 1) % 5 == 0 or i + 1 == len(ts):
                pickle.dump(cache, open(cf, "wb")); print(f"{name}: {i+1}/{len(ts)} emails, {(time.time()-t0)/60:.1f} min", flush=True)
        for i, out in cache.items():
            for vn, v in out.items(): per_rows.append(dict(model=name, email=i, variant=vn, **v))
    P = pd.DataFrame(per_rows); save(P, "ig_controls_per_email.csv")
    summ = []
    for (m, vn), g in P.groupby(["model", "variant"]):
        r = dict(model=m, variant=vn, n=len(g), comp=g.comp.mean(), comp_sd=g.comp.std(), suff=g.suff.mean(), suff_sd=g.suff.std())
        for col in ("h_comp", "h_suff", "delta", "spearman_lime"):
            if col in g and g[col].notna().any(): r[col] = g[col].mean()
        rnd = P[(P.model == m) & (P.variant == "Random")].sort_values("email").comp.values
        lim = P[(P.model == m) & (P.variant == "LIME")].sort_values("email").comp.values
        mine = g.sort_values("email").comp.values
        if vn not in ("Random",):
            r["p_vs_random"] = stats.wilcoxon(mine, rnd).pvalue if len(mine) == len(rnd) else np.nan
        if vn not in ("LIME", "Random"):
            r["p_vs_lime"] = stats.wilcoxon(mine, lim).pvalue
        summ.append(r)
    S = pd.DataFrame(summ); save(S, "ig_controls_summary.csv"); print(S.round(4).to_string())

# =====================================================================================
if "shap" in STEPS:
    print("\n=== shap: budget sensitivity (same emails, 300 / 1000 / 2000 evaluations) ===")
    D = data(); tr, va, te = D["split"]; rows = []
    for name in MODELS:
        model, tok = load_model(name); ts = pick_emails(te, model, tok, a.n_shap, 1)
        cf = os.path.join(OUT, f"shap_cache_{name}.pkl"); cache = pickle.load(open(cf, "rb")) if os.path.exists(cf) else {}
        for i, tx in enumerate(ts):
            if i in cache: continue
            vocab = vocab_of(tx); rec = {}
            for ev in (300, 1000, 2000):
                t0 = time.time(); sc = exp_shap(model, tok, tx, vocab, ev); c, s = faith(*curves(model, tok, tx, sc))
                rec[ev] = dict(comp=c, suff=s, sec=time.time() - t0)
            cache[i] = rec
            if (i + 1) % 5 == 0: pickle.dump(cache, open(cf, "wb")); print(name, i + 1, "/", len(ts), flush=True)
        pickle.dump(cache, open(cf, "wb"))
        for ev in (300, 1000, 2000):
            c = [cache[i][ev]["comp"] for i in cache]; s_ = [cache[i][ev]["suff"] for i in cache]; t_ = [cache[i][ev]["sec"] for i in cache]
            rows.append(dict(model=name, evals=ev, n=len(c), comp=np.mean(c), suff=np.mean(s_), sec_per_email=np.mean(t_)))
    save(pd.DataFrame(rows), "shap_budget.csv"); print(pd.DataFrame(rows).round(4).to_string())

# =====================================================================================
if "seeds" in STEPS:
    print("\n=== seeds: RoBERTa-base seeds 43 and 44 (dedup) ===")
    torch, dev = torch_env()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    D = data(); rows = []
    for seed in (43, 44):
        cf = ckf("train", "dedup", seed, "roberta-base") + ".pkl"
        tr, va, te = split(D["dedup"], seed)
        if os.path.exists(cf): prob = pickle.load(open(cf, "rb"))["prob"]; print("resumed", seed)
        else:
            def sd(s):
                random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
            sd(seed); tok = AutoTokenizer.from_pretrained("roberta-base")
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(dev)
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
                print("seed", seed, "epoch", ep + 1, "val f1", round(f1, 4), flush=True)
                if f1 > best: best = f1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state); model.eval(); prob = predict_proba(model, tok, te.text)
            pickle.dump(dict(prob=prob, log=None), open(cf, "wb")); del model; torch.cuda.empty_cache()
        rows.append(dict(seed=seed, **metrics(te.y.values, (prob >= .5).astype(int))))
    save(pd.DataFrame(rows), "roberta_extra_seeds.csv"); print(pd.DataFrame(rows).round(4).to_string())

# ---------------- zip ----------------
zp = os.path.join(FULL, "controls_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(OUT):
        if not f.endswith(".pkl"): z.write(os.path.join(OUT, f), f)
print("\nDone. Send me:", zp)
try:
    from google.colab import files; files.download(zp)
except Exception: pass
