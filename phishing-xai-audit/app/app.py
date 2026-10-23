"""Phishing-email detector demo with word-level explanations.

Companion demo for "Beyond Accuracy: Auditing Duplicates, Baselines and Explanation Faithfulness in
Transformer-Based Phishing Email Detection". It loads a fine-tuned RoBERTa-style checkpoint, classifies an
email as phishing or safe, and shows which words support or oppose the prediction according to LIME or
Integrated Gradients (IG). It also runs a small deletion check, the same idea as the paper's
comprehensiveness test: delete the top-k words and see how much the predicted-class probability drops.

Run:
    pip install -r requirements.txt
    MODEL_DIR=/path/to/checkpoint python app.py          # a folder saved with save_pretrained()
    MODEL_ID=your-hf-username/your-model python app.py   # or a Hugging Face Hub id (set HF_TOKEN if private)

The label convention of the paper's models is 1 = phishing, 0 = safe. Weights are NOT included in this
repository. This is a research demo, not a security product.
"""
import os
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 256                      # the transformers in the paper read only the first 256 tokens
LABELS = {0: "safe", 1: "phishing"}
WORD_RE = re.compile(r"[A-Za-z0-9]+")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = ["LIME", "IG (pad baseline, probability target: paper default)", "IG (mask baseline, logit target)"]

_model = _tok = None


def load(path_or_id=None):
    """Load the checkpoint once (also lets tests inject a tiny model)."""
    global _model, _tok
    src = path_or_id or os.environ.get("MODEL_DIR") or os.environ.get("MODEL_ID")
    if not src:
        raise RuntimeError("Set MODEL_DIR (a save_pretrained folder) or MODEL_ID (a Hugging Face Hub id).")
    _tok = AutoTokenizer.from_pretrained(src)
    _model = AutoModelForSequenceClassification.from_pretrained(src).to(DEVICE).eval()
    return _model, _tok


# ----------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_proba(texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _tok(list(texts[i:i + batch_size]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        out.append(torch.softmax(_model(**enc).logits, -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2))


# ----------------------------------------------------------------------------- explanations
def words_of(text):
    """(word, start, end) for every alphanumeric word, in order."""
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def explain_lime(text, target, num_samples=500, seed=0):
    from lime.lime_text import LimeTextExplainer
    exp = LimeTextExplainer(class_names=[LABELS[0], LABELS[1]], split_expression=r"[^A-Za-z0-9]+", bow=True, random_state=seed)
    e = exp.explain_instance(text, lambda xs: predict_proba(xs), labels=(target,), num_features=max(1, len(set(w.lower() for w, _, _ in words_of(text)))), num_samples=num_samples)
    weights = {w.lower(): s for w, s in e.as_list(label=target)}
    return [(w, weights.get(w.lower(), 0.0)) for w, _, _ in words_of(text)]


def explain_ig(text, target, baseline="pad", objective="prob", steps=None):
    """Layer Integrated Gradients at the embedding layer; token attributions are summed into words."""
    from captum.attr import LayerIntegratedGradients
    steps = steps or (30 if objective == "prob" else 100)
    enc = _tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(DEVICE); mask = enc["attention_mask"].to(DEVICE)
    special = torch.tensor(_tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True), dtype=torch.bool, device=DEVICE)
    fill = _tok.pad_token_id if baseline == "pad" else _tok.mask_token_id
    base = torch.where(special.unsqueeze(0), ids, torch.full_like(ids, fill))

    def forward(input_ids, attention_mask):
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
        if objective == "prob":
            return torch.softmax(logits, -1)[:, target]
        return logits[:, target] - logits[:, 1 - target]

    lig = LayerIntegratedGradients(forward, _model.get_input_embeddings())
    with torch.enable_grad():
        attr = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), n_steps=steps, internal_batch_size=16)
    tok_scores = attr.sum(-1)[0].detach().cpu().numpy()
    words = words_of(text)
    scores = np.zeros(len(words))
    for (s, e), a in zip(offsets, tok_scores):
        if e <= s:
            continue                                   # special tokens
        for k, (_, ws, we) in enumerate(words):
            if s < we and e > ws:                      # token overlaps word k
                scores[k] += a
                break
    return [(w, float(sc)) for (w, _, _), sc in zip(words, scores)]


def explain(text, method, target, num_samples=500):
    if method == "LIME":
        return explain_lime(text, target, num_samples)
    if method.startswith("IG (pad"):
        return explain_ig(text, target, "pad", "prob")
    return explain_ig(text, target, "mask", "logit")


# ----------------------------------------------------------------------------- deletion check
def delete_words(text, drop):
    drop = {w.lower() for w in drop}
    return re.sub(r"\s+", " ", WORD_RE.sub(lambda m: "" if m.group().lower() in drop else m.group(), text)).strip()


def deletion_check(text, scored, target, ks=(1, 3, 5, 10)):
    """Comprehensiveness: p(target | original) - p(target | top-k supporting words deleted), same class throughout."""
    best = {}
    for w, s in scored:                                # one score per distinct word, highest wins
        best[w.lower()] = max(best.get(w.lower(), -np.inf), s)
    ranked = [w for w, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    p0 = float(predict_proba([text])[0, target])
    rows = []
    for k in ks:
        if k > len(ranked):
            break
        pk = float(predict_proba([delete_words(text, ranked[:k])])[0, target])
        rows.append([k, round(p0, 4), round(pk, 4), round(p0 - pk, 4)])
    return rows


# ----------------------------------------------------------------------------- one call for the UI
def analyze(text, method, num_samples=500):
    text = (text or "").strip()
    if not text:
        return {}, [], [], [], "Paste an email first."
    probs = predict_proba([text])[0]
    target = int(probs.argmax())
    scored = explain(text, method, target, int(num_samples))
    peak = max((abs(s) for _, s in scored), default=0.0) or 1.0            # abs-max normalisation
    # highlight in the original text: keep the characters between words as plain text
    spans, pos = [], 0
    for (w, s), (_, ws, we) in zip(scored, words_of(text)):
        if ws > pos:
            spans.append((text[pos:ws], None))
        spans.append((w, round(s / peak, 3)))
        pos = we
    if pos < len(text):
        spans.append((text[pos:], None))
    best = {}
    for w, s in scored:
        if abs(s) > abs(best.get(w.lower(), 0.0)):
            best[w.lower()] = s
    top = [[w, round(s / peak, 3)] for w, s in sorted(best.items(), key=lambda kv: -kv[1])[:10]]
    n_tokens = len(_tok(text, truncation=False)["input_ids"])
    note = (f"Explained class: {LABELS[target]}. Positive scores support this prediction, negative scores oppose it. "
            f"The model reads only the first {MAX_LEN} tokens (this email has {n_tokens}). "
            "Per the paper, word-level explanations depend on the method and its settings: treat them as hints about the model, not as proof.")
    return {LABELS[i]: float(p) for i, p in enumerate(probs)}, spans, top, deletion_check(text, scored, target), note


EXAMPLES = [
    ["Dear customer, we detected unusual activity on your account. Your access will be suspended within 24 hours unless you verify your password and card details at http://example.test/verify-now. Act immediately."],
    ["Hi team, attached are the minutes from Tuesday's project meeting. Please review the action items and send me corrections by Friday. Thanks, Priya"],
]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Phishing email detector with explanations") as demo:
        gr.Markdown("# Phishing email detector with word-level explanations\n"
                    "Research demo for the paper *Beyond Accuracy: Auditing Duplicates, Baselines and Explanation Faithfulness in "
                    "Transformer-Based Phishing Email Detection*. Not a security product; never paste confidential email.")
        with gr.Row():
            with gr.Column():
                box = gr.Textbox(lines=10, label="Email text", placeholder="Paste an email here")
                method = gr.Dropdown(METHODS, value=METHODS[0], label="Explanation method")
                n = gr.Slider(100, 1000, value=500, step=100, label="LIME samples (more = steadier, slower)")
                btn = gr.Button("Analyze", variant="primary")
                gr.Examples(EXAMPLES, inputs=[box])
            with gr.Column():
                label = gr.Label(label="Prediction")
                hl = gr.HighlightedText(label="Words supporting (+) or opposing (-) the prediction", show_legend=False, combine_adjacent=False)
                note = gr.Markdown()
        with gr.Row():
            top = gr.Dataframe(headers=["word", "score (abs-max normalised)"], label="Top 10 words supporting the prediction", interactive=False)
            dele = gr.Dataframe(headers=["k", "p(original)", "p(after deleting top-k words)", "drop"], label="Deletion check on the predicted class (a larger drop means the explanation is more faithful)", interactive=False)
        btn.click(analyze, [box, method, n], [label, hl, top, dele, note])
    return demo


if __name__ == "__main__":
    load()
    build_ui().queue().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", "7860")))
