# Beyond Accuracy: Auditing Duplicates, Baselines and Explanation Faithfulness in Transformer-Based Phishing Email Detection

Code, saved results, paper source and a demo app for our IEEE conference paper. The paper builds on Uddin et al. (2026),
*An explainable transformer-based model for phishing email detection*, Computer Networks 277, 112061, and re-examines
what a headline accuracy number on a public phishing-email corpus really tells us.

**Authors:** Tellakula Muneiah, Chandra Akhil, Nandulamatam Hemitha, Chagaleti Tejeswar
(Department of Computer Science and Engineering, RGMCET, Andhra Pradesh, India)

> Project page: `https://YOUR-USERNAME.github.io/phishing-xai-audit/` (after you enable GitHub Pages, see below)
> Repository: `https://github.com/YOUR-USERNAME/phishing-xai-audit`

## What the paper asks

1. **Duplicates and label noise.** How much of the ~98–99% accuracy on the Kaggle phishing-email corpus survives
   de-duplication, and what do conflicting duplicates (the "empty" placeholder) do to the ceiling?
2. **Baselines.** Do transformers beat TF-IDF baselines once the baselines see the same first 256 tokens and are
   tuned on validation data, with paired tests and multiple-comparison correction?
3. **Faithfulness.** Do LIME, SHAP and Integrated Gradients (and a LIME+IG hybrid) agree, and do their explanations
   pass ERASER comprehensiveness/sufficiency checks with the predicted class held fixed?
4. **Robustness and stability.** How do the models behave under homoglyph, zero-width and typo obfuscation, and how
   stable are the explanations when only the random seed of the explainer changes?

Headline numbers (dedup protocol, mean ± SD over seeds, accuracy %, Table II of the paper):

| Model | Accuracy |
|---|---|
| DistilRoBERTa | 98.58 ± 0.23 |
| RoBERTa-base | 98.84 ± 0.13 |
| TF-IDF + SVM | 98.65 ± 0.16 |
| TF-IDF + LR | 98.59 ± 0.20 |

The differences are small, and the paper says so. Read the limitations section of the paper before quoting any of this.

## Repository layout

```
notebooks/   Colab notebooks (run on a free T4 GPU)
  01_main_experiments.ipynb       main run: baselines, transformers, explainers, faithfulness, robustness
  02_round2_experiments.ipynb     follow-up checks: tuned/matched baselines, near-duplicates, "empty" sensitivity,
                                  LIME repeats, stratified explanation sample, RoBERTa re-seeds
  superseded/                     an old notebook NOT used by the paper (see its README)
scripts/     the same code as stand-alone scripts (called by the notebooks)
  controls_colab.py               paired tests, overlap, robustness length controls, IG controls, SHAP budget
  audit_checks.py                 duplicate/leakage diagnostics, per-email significance tests, Cyrillic prevalence
  round2_colab.py                 the round-2 steps listed above
results/     every CSV/JSON the paper's numbers and figures are computed from
  main_run_r6/                    main run (run 2): the paper's tables
  replication_run_r5/             earlier full run (run 1), kept as a replication
  controls_ctl/                   control experiments
  round2_r2res/                   round-2 experiments
  audit_out.txt                   output of audit_checks.py
paper/       the paper (.docx and a preview PDF) and the code that builds it from results/
app/         Gradio demo (prediction + LIME/IG highlights + deletion check); weights not included
docs/        project web page for GitHub Pages
```

## Reproduce

**Data (not included).** Download the Kaggle dataset *Phishing Email Dataset* (`subhajournal/phishingemails`), file
`Phishing_Email.csv`, and put it in your Google Drive at `MyDrive/phish_xai/Phishing_Email.csv`. The external test
samples (Enron-Spam, zefang-liu) are pulled from Hugging Face by the notebooks. The datasets have their own licences
and are not redistributed here.

Run in this order on Colab (Runtime → T4 GPU; each step checkpoints to Drive and can be resumed):

1. `notebooks/01_main_experiments.ipynb` – trains/evaluates everything and saves checkpoints to
   `MyDrive/phish_xai/full/ckpt/`.
2. `scripts/controls_colab.py` – needs step 1's outputs and checkpoints.
3. `scripts/audit_checks.py` – CPU only.
4. `notebooks/02_round2_experiments.ipynb` – needs steps 1–2. Its optional cell 7 retrains RoBERTa-base with seeds 43
   and 44 (the seeds the paper reports).
5. Rebuild the paper from the saved results (no GPU needed):

```bash
cd paper/build
pip install -r requirements.txt
python3 make_figs.py
python3 -I build_paper.py ../Phishing_XAI_IEEE_Paper.docx
```

`build_paper.py` reads `../../results`; set `PAPER_DATA=/path/to/results` to point it elsewhere. The `.docx` is built on
the supplied IEEE template styles (`paper/build/template/v2.docx`). The PDF in `paper/` is only a LibreOffice preview;
export the final PDF from Word and check the page limit yourself.

## Which results folder is which

- `results/main_run_r6` is **run 2**, the main run behind the paper's tables. `results/replication_run_r5` is run 1;
  the two may differ slightly (GPU training is not bit-reproducible).
- The RoBERTa-base seeds 43 and 44 in the paper come from `results/round2_r2res/roberta_reseeds.csv`. The older
  `results/controls_ctl/roberta_extra_seeds.csv` is superseded: that notebook ran validation with dropout still active.

## Limitations (also stated in the paper)

- Three seeds for most results, one seed for the near-duplicate group splits with transformers; this supports
  descriptive statements, not strong claims.
- LIME re-sampling variation and between-pipeline-run variation are measured, but sample and checkpoint effects are not
  separated.
- The explanation analyses use a sample (100 emails, and a stratified sample of 120), not the whole test set.
- A single public corpus plus two small external samples.
- Per-email predictions of the transformers on the raw split were not saved, so the identity of their errors on the
  "empty" rows could not be checked.

## Model weights and the demo

Weights are not committed (`.gitignore` blocks them). The seed-42 checkpoints produced by notebook 01 sit in the
authors' Google Drive under `phish_xai/full/ckpt/`. See `app/README.md` for running the demo with them.

## Licence and citation

Code: MIT (see `LICENSE`; datasets are not covered). Citation metadata is in `CITATION.cff`.

## Put it on GitHub

1. On github.com click **New repository**, name it `phishing-xai-audit`, leave it empty (no README).
2. Upload this folder's contents (drag and drop in the browser works; files over 25 MB need git), or from a terminal:

```bash
cd phishing-xai-audit
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/phishing-xai-audit.git
git push -u origin main
```

3. Optional project page: **Settings → Pages → Deploy from a branch → `main` / `docs`**. After a minute it is live.
4. Replace `YOUR-USERNAME` in this file and in `docs/index.html`. For one-click Colab, the link is
   `https://colab.research.google.com/github/YOUR-USERNAME/phishing-xai-audit/blob/main/notebooks/01_main_experiments.ipynb`.
