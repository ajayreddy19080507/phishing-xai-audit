---
title: Phishing Email Detector with Explanations
emoji: 🎣
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Demo app: phishing-email detector with word-level explanations

A small Gradio app for the paper *Beyond Accuracy: Auditing Duplicates, Baselines and Explanation Faithfulness in
Transformer-Based Phishing Email Detection*. Paste an email and the app shows

- the prediction (phishing or safe) with its probability,
- the words that support (+) or oppose (−) the prediction according to **LIME** or **Integrated Gradients**. There are
  two IG settings, because the paper shows that the pad-baseline default and a mask-baseline/logit setting can disagree,
- a **deletion check**: the probability of the predicted class after deleting the top-1, 3, 5 and 10 supporting words
  (the paper's comprehensiveness idea). A bigger drop means the explanation is more faithful to the model.

It is a research demo, not a security product. Do not paste confidential email.

## What you need to supply

The model weights are **not** in this repository. You need a fine-tuned checkpoint with label 1 = phishing and
0 = safe, for example the DistilRoBERTa or RoBERTa-base model that `notebooks/01_main_experiments.ipynb` saves to
Google Drive (`phish_xai/full/ckpt/model_distilroberta-base` or `.../model_roberta-base`). Download that folder to
your computer.

## Run locally

```bash
pip install -r requirements.txt
MODEL_DIR=/path/to/model_distilroberta-base python app.py
```

On Windows PowerShell:

```powershell
$env:MODEL_DIR="C:\path\to\model_distilroberta-base"; python app.py
```

Open http://127.0.0.1:7860. On a CPU, LIME with 500 samples takes about a minute for RoBERTa-base and less for
DistilRoBERTa; IG is faster. Use the slider to trade speed for steadier LIME scores.

## Host it for free on a Hugging Face Space (optional)

1. Create a **private model** repository on huggingface.co and upload the checkpoint folder to it.
2. Create a new **Space** (SDK: Gradio) and upload `app.py`, `requirements.txt` and this `README.md`.
3. In the Space settings add the variable `MODEL_ID` = `your-username/your-model` and, because the model is private,
   a secret `HF_TOKEN`.

## Notes and limits

- Like the paper's models, it reads only the first 256 tokens; the note under the prediction tells you if the email is longer.
- Scores sit on whole words (letters and digits). Punctuation and URL symbols carry attribution inside the model but are not shown,
  so the shown IG scores need not sum to the full probability change.
- Scores are abs-max normalised per email, so they are comparable within an email, not across emails.
- The code was tested end to end on a tiny locally trained RoBERTa-style model (prediction, LIME, both IG settings,
  the deletion check, the UI call, and the IG completeness property on text without punctuation). It has not been run
  against the trained checkpoints, so check one email you understand before showing it to anyone.
