"""Every number that appears in paper v3 is computed here from the result files."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

W = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PAPER_DATA") or os.path.normpath(os.path.join(W, "..", "..", "results"))
R2 = f"{DATA}/main_run_r6"      # "run 2": main run (weights retained, controls run on it)
R1 = f"{DATA}/replication_run_r5"      # "run 1": identical code and seeds, earlier training run
CT = f"{DATA}/controls_ctl"
MODELS = ["distilroberta-base", "roberta-base"]
METHODS = ["LIME", "SHAP", "IG", "LITA-minmax(a=.5)", "LITA-absmax(a=.5)", "LITA-absmax(tuned)", "Random"]


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); out[i] = min(1, run)
    return out


def wil(x, y):
    try: return float(stats.wilcoxon(x, y).pvalue)
    except Exception: return 1.0


def load():
    N = {}
    meta = json.load(open(f"{R2}/meta.json")); N["meta"] = meta; a = meta["audit"]; N["audit"] = a
    allr = pd.read_csv(f"{R2}/all_results.csv"); N["all"] = allr
    allr1 = pd.read_csv(f"{R1}/all_results.csv"); N["all1"] = allr1
    # summary_dedup has a two-row header
    sd = pd.read_csv(f"{R2}/summary_dedup.csv"); sd = sd.iloc[2:].set_index(sd.columns[0]); N["sd"] = sd.astype(float)
    N["ci"] = pd.read_csv(f"{R2}/f1_bootstrap_ci.csv").set_index("model")
    N["mcn"] = pd.read_csv(f"{R2}/mcnemar.csv")
    N["paired"] = pd.read_csv(f"{CT}/paired_accuracy_diff.csv")
    N["rob"] = pd.read_csv(f"{R2}/robustness.csv")
    N["rc"] = pd.read_csv(f"{CT}/robust_controls.csv")
    N["ext"] = pd.read_csv(f"{CT}/external_overlap.csv").set_index("dataset")
    N["xd"] = pd.read_csv(f"{R2}/cross_dataset.csv")
    N["faith_tab"] = pd.read_csv(f"{R2}/explanation_faithfulness.csv")
    N["faith_tab1"] = pd.read_csv(f"{R1}/explanation_faithfulness.csv")
    N["shap"] = pd.read_csv(f"{CT}/shap_budget.csv")
    ag = pd.read_csv(f"{R2}/explanation_agreement.csv"); ag = ag.iloc[2:].set_index(ag.columns[0]).astype(float); N["ag"] = ag
    pe = pd.read_csv(f"{CT}/per_email_faithfulness.csv"); N["pe"] = pe
    igp = pd.read_csv(f"{CT}/ig_controls_per_email.csv"); N["igp"] = igp
    # ---- per-email statistics (bootstrap CIs, Holm-corrected Wilcoxon)
    rng = np.random.RandomState(0); F = {}
    pairs = [("LITA-absmax(tuned)", "LIME"), ("LITA-absmax(tuned)", "IG"), ("LITA-absmax(tuned)", "SHAP"),
             ("LITA-absmax(a=.5)", "LIME"), ("LITA-absmax(a=.5)", "IG"), ("LITA-absmax(a=.5)", "SHAP"),
             ("LIME", "SHAP"), ("LIME", "IG"), ("SHAP", "IG"), ("LIME", "Random"), ("SHAP", "Random"),
             ("IG", "Random"), ("LITA-absmax(a=.5)", "Random")]
    for m in MODELS:
        s = pe[pe.model == m]
        for met in ("comp", "suff"):
            D = {k: s[s.method == k].sort_values("email")[met].values for k in METHODS}
            F[(m, met, "mean")] = {k: float(D[k].mean()) for k in METHODS}
            F[(m, met, "ci")] = {}
            for k in METHODS:
                bs = [D[k][rng.randint(0, len(D[k]), len(D[k]))].mean() for _ in range(2000)]
                F[(m, met, "ci")][k] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            ps = [wil(D[x], D[y]) for x, y in pairs]; adj = holm(ps)
            F[(m, met, "p")] = {pr: (p, q) for pr, p, q in zip(pairs, ps, adj)}
        F[(m, "n")] = int(s[s.method == "LIME"].shape[0])
        # first 30 emails (same emails used in the SHAP-budget control)
        s30 = s[s.email < 30]
        F[(m, "first30")] = {k: float(s30[s30.method == k].comp.mean()) for k in ("LIME", "SHAP", "IG", "Random")}
    N["F"] = F
    # ---- IG controls
    G = {}
    for m in MODELS:
        L = igp[(igp.model == m) & (igp.variant == "LIME")].sort_values("email")
        R = igp[(igp.model == m) & (igp.variant == "Random")].sort_values("email")
        for v in igp.variant.unique():
            g = igp[(igp.model == m) & (igp.variant == v)].sort_values("email")
            r = dict(comp=g.comp.mean(), suff=g.suff.mean())
            r["ci"] = tuple(np.percentile([g.comp.values[rng.randint(0, len(g), len(g))].mean() for _ in range(2000)], [2.5, 97.5]))
            if v != "Random": r["p_rand"] = wil(g.comp.values, R.comp.values)
            if v not in ("Random", "LIME"): r["p_lime"] = wil(g.comp.values, L.comp.values)
            if v.startswith(("IG", "Grad")):
                r["h_comp"] = g.h_comp.mean(); r["h_suff"] = g.h_suff.mean()
                r["h_p_lime"] = wil(g.h_comp.values, L.comp.values)
                r["rho"] = g.spearman_lime.mean(); r["rho_sd"] = g.spearman_lime.std()
                if g.delta.notna().any(): r["delta_med"] = g.delta.median(); r["delta_mean"] = g.delta.mean()
            G[(m, v)] = r
    N["G"] = G
    return N


if __name__ == "__main__":
    N = load()
    for k, v in N["G"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
