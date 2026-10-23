"""
Pre-submission diagnostic checks for the phishing-XAI paper.
Run in the same Colab/Drive setup as the main notebook (CPU is enough).

    DRIVE = "/content/drive/MyDrive/phish_xai"      # change if needed
    !python audit_checks.py --csv /content/drive/MyDrive/phish_xai/Phishing_Email.csv --drive /content/drive/MyDrive/phish_xai

Part A  - duplicate / leakage diagnostics (needs Phishing_Email.csv)
Part B  - per-email significance tests for faithfulness (needs the Drive checkpoints ckpt/expl_*.pkl)
Part C  - Cyrillic prevalence by class (explains homoglyph false positives?)
Nothing here trains a transformer.
"""
import argparse, os, pickle, re, json, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="Phishing_Email.csv")
ap.add_argument("--drive", default="/content/drive/MyDrive/phish_xai")
ap.add_argument("--skip-a", action="store_true")
ap.add_argument("--skip-b", action="store_true")
a = ap.parse_args()

# ---- identical helpers to the notebook -------------------------------------------------
def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t));  return re.sub(r"\s+", " ", t).strip()
def key(t):
    return re.sub(r"\W+", " ", str(t).lower()).strip()
def split(df, seed):
    tr, te = train_test_split(df, test_size=0.20, stratify=df.y, random_state=seed)
    tr, va = train_test_split(tr, test_size=0.125, stratify=tr.y, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def load(path):
    df = pd.read_csv(path); df.columns = [c.strip() for c in df.columns]
    tcol = [c for c in df.columns if "text" in c.lower()][0]
    lcol = [c for c in df.columns if "type" in c.lower() or "label" in c.lower()][0]
    df = df[[tcol, lcol]].dropna(); df.columns = ["text", "label_raw"]
    df["text"] = df["text"].map(clean); df = df[df.text.str.len() > 0].reset_index(drop=True)
    df["y"] = df.label_raw.astype(str).str.lower().str.contains("phish").astype(int)
    df["key"] = df.text.map(key); return df

# =========================================================================================
if not a.skip_a:
    print("\n=== PART A: duplicate groups =================================================")
    raw = load(a.csv)
    g = raw.groupby("key").agg(n=("y", "size"), n_phish=("y", "sum"), prev=("text", lambda s: s.iloc[0][:90]))
    g["conflict"] = (g.n_phish > 0) & (g.n_phish < g.n)
    dg = g[g.n > 1].sort_values("n", ascending=False)
    print("groups with >1 copy:", len(dg), "| rows in them:", int(dg.n.sum()))
    hist = pd.cut(dg.n, [1, 2, 3, 5, 10, 50, 10**6]).value_counts().sort_index()
    print("group-size histogram (copies per group):", {str(k): int(v) for k, v in hist.items()})
    print("empty normalised key present:", (raw.key == "").sum(), "rows")
    print("\n10 largest groups (n, n_phish, conflict, preview):")
    print(dg.head(10)[["n", "n_phish", "conflict", "prev"]].to_string())
    print("\nconflicting groups:"); print(dg[dg.conflict][["n", "n_phish", "prev"]].to_string())

    print("\n=== PART A2: seen/unseen accuracy for TF-IDF LR on the raw seed-42 split (CPU) ====")
    tr, va, te = split(raw, 42)
    seen = te.key.isin(set(tr.key)).values
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text)
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xtr, tr.y)
    pr = clf.predict(Xte); y = te.y.values
    conf_keys = set(dg[dg.conflict].index)
    in_conf = te.key.isin(conf_keys).values
    # does the train copy carry the same label?
    lab = tr.groupby("key").y.agg(lambda s: s.mode().iloc[0])
    te["train_label"] = te.key.map(lab)
    same = (te.train_label.values == y)
    def rep(name, m):
        if m.sum(): print(f"{name:55s} n={int(m.sum()):5d}  acc={(pr[m]==y[m]).mean():.4f}")
    rep("all test", np.ones(len(te), bool)); rep("seen in train", seen); rep("unseen", ~seen)
    rep("seen, in the conflicting group(s)", seen & in_conf)
    rep("seen, NOT in conflicting group", seen & ~in_conf)
    rep("seen, train label == test label", seen & same)
    rep("seen, train label != test label", seen & ~same)
    print("seen emails by length quartile (words):")
    wl = te.text.str.split().str.len().values
    for lo, hi in [(0, 20), (20, 100), (100, 400), (400, 10**9)]:
        rep(f"  seen & {lo}-{hi} words", seen & (wl >= lo) & (wl < hi))

    print("\n=== PART A3: near-duplicates (TF-IDF cosine >= 0.95) test vs train, not exact ===")
    v1 = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(tr.text)
    A = v1.transform(tr.text); B = v1.transform(te.text)
    best = np.zeros(B.shape[0])
    for i in range(0, B.shape[0], 500):
        best[i:i+500] = (B[i:i+500] @ A.T).max(axis=1).toarray().ravel()
    near = (best >= 0.95) & ~seen
    print("test emails with a near-duplicate (>=0.95) in train but no exact duplicate:", int(near.sum()),
          f"({near.mean():.3%})")
    rep("  accuracy on those", near)

# =========================================================================================
if not a.skip_b:
    print("\n=== PART B: per-email significance of faithfulness (from Drive checkpoints) ======")
    ck = os.path.join(a.drive, "full", "ckpt")
    METHODS = ["LIME", "SHAP", "IG", "LITA-minmax(a=.5)", "LITA-absmax(a=.5)", "LITA-absmax(tuned)", "Random"]
    def holm(ps):
        o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
        for r, i in enumerate(o):
            run = max(run, (m - r) * ps[i]); out[i] = min(1, run)
        return out
    rng = np.random.RandomState(0)
    for name in ["distilroberta-base", "roberta-base"]:
        f = os.path.join(ck, f"expl_{name}.pkl")
        if not os.path.exists(f): print("missing", f); continue
        recs = pickle.load(open(f, "rb"))
        C = {m: np.array([r["res"][m][0] for r in recs]) for m in METHODS}
        S = {m: np.array([r["res"][m][1] for r in recs]) for m in METHODS}
        print(f"\n--- {name}: n={len(recs)} emails")
        for lbl, D in [("COMPREHENSIVENESS (higher better)", C), ("SUFFICIENCY (lower better)", S)]:
            print(lbl)
            for m in METHODS:
                bs = [D[m][rng.randint(0, len(D[m]), len(D[m]))].mean() for _ in range(2000)]
                print(f"  {m:22s} mean={D[m].mean():.4f}  95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]"
                      f"  median={np.median(D[m]):.4f}")
            pairs = [("LITA-absmax(a=.5)", "LIME"), ("LITA-absmax(a=.5)", "IG"), ("LITA-absmax(a=.5)", "SHAP"),
                     ("LITA-minmax(a=.5)", "LIME"), ("LITA-minmax(a=.5)", "IG"),
                     ("LIME", "SHAP"), ("LIME", "IG"), ("SHAP", "IG"),
                     ("LIME", "Random"), ("SHAP", "Random"), ("IG", "Random"), ("LITA-absmax(a=.5)", "Random")]
            ps = []
            for x, y_ in pairs:
                try: ps.append(stats.wilcoxon(D[x], D[y_]).pvalue)
                except Exception: ps.append(1.0)
            adj = holm(np.array(ps))
            for (x, y_), p, q in zip(pairs, ps, adj):
                print(f"  Wilcoxon {x:20s} vs {y_:8s}: p={p:.2e}  Holm-adj p={q:.2e}")

# =========================================================================================
print("\n=== PART C: Cyrillic prevalence by class (training corpus) =========================")
if not a.skip_a:
    cy = raw.text.map(lambda t: bool(re.search(r"[Ѐ-ӿ]", t)))
    print(pd.crosstab(raw.y.map({0: "safe", 1: "phish"}), cy, margins=True))
    nonascii = raw.text.map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))
    print("mean share of non-ASCII chars:", nonascii.groupby(raw.y).mean().round(5).to_dict())
