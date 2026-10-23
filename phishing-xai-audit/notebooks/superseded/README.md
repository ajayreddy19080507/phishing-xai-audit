# Superseded notebook

`roberta_extra_seeds_colab.ipynb` trained RoBERTa-base seeds 43 and 44, but its validation-epoch selection ran with
dropout still active (no `model.eval()` in the probability function), so the "best epoch" could be chosen on noisy
scores. The paper does **not** use these seeds. The RoBERTa-base seeds 43 and 44 reported in the paper come from the
`reseeds` step of `scripts/round2_colab.py` (notebook `02_round2_experiments.ipynb`, optional cell 7), which validates
in eval mode. The old output is kept only as `results/controls_ctl/roberta_extra_seeds.csv` for transparency.
