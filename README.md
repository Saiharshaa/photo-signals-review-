# Review triage: sentiment, issue detection and photo signals

BC3415 Individual Assignment, Option A.

A phone-accessories retailer receives thousands of reviews a day, as text and as customer photos. This project:
1. classifies each review as **positive / neutral / negative**,
2. tags **what went wrong** (defect/dead on arrival, counterfeit, wrong item, damaged in shipping, performance), and
3. combines text, metadata and photo signals into a **ranked queue of reviews that point to problematic orders**.

## Files

| Path | What it is |
|---|---|
| `review_triage.ipynb` | The whole analysis: audit → preprocessing → models → explainability → triage + images → Gradio demo → limitations |
| `triage.py` | Reusable module: text cleaning, keyword issue tagger, flag score. Weights and severities are at the top of the file |
| `scripts/build_sample.py` | Rebuilds `data/sample.csv` from the source dataset (HTTP range sampling, seed 42) |
| `data/sample.csv` | 9,000 reviews, 3,000 per class, with customer image URLs |
| `data/pool_stats.json` | Natural class mix of the 231,137-review pool the sample was drawn from |
| `reports/results.csv` | Model comparison table (written by the notebook) |
| `reports/flag_queue.csv` | Top-20 flagged reviews (written by the notebook) |
| `video/script.md` | Narration for the 4-minute walkthrough |
| `video/record.py` | Optional: records a scrolling walkthrough on the script's timings and adds your voiceover |

## Setup and run

Requires Python 3.11 and runs on CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter notebook review_triage.ipynb   # then Kernel > Restart & Run All
```

- Restart & Run All took **5.2 minutes** (kernel time) on a 12-thread laptop CPU. BERT inference on the test set is about 3 minutes of that.
- The first run also downloads two pretrained models (~1.3 GB) and 100 customer photos (cached in `data/images/`).
- The last cell launches a Gradio demo with a temporary public `share=True` link.
- Optional: `python scripts/build_sample.py` re-downloads the sample (~100 MB of HTTP range requests, ~1 minute).

**Video (optional).** Run `pip install playwright imageio-ffmpeg` and `playwright install chromium`, put your voiceover at `video/voice.m4a`, then run `python video/record.py`.
- It exports the notebook to HTML, records Chromium scrolling through each section on the `video/script.md` timings, and merges the voice into `video/walkthrough.mp4`.
- The live Gradio demo has to be screen-recorded separately.

## Data

**Amazon Reviews 2023** (McAuley Lab, UCSD), category `Cell_Phones_and_Accessories`, from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023.

The full category file is 9.3 GB and sorted by user. `build_sample.py` reads 400 random 256 KB byte ranges across the whole file, which gives a 231,137-review pool. It then draws 3,000 reviews per class: 1–2★ negative, 3★ neutral, 4–5★ positive.

| | Negative | Neutral | Positive |
|---|---|---|---|
| Natural share in the pool | 19.2% | 7.3% | 73.5% |
| Balanced sample | 3,000 | 3,000 | 3,000 |
| With customer photos | 7.6% | 4.6% | 4.2% |

**Licence/terms:** neither the dataset card nor the project site states an explicit licence. The authors ask for the citation below, and the data is used here for academic coursework only.

**Data-quality findings** (details in notebook §1–2):
- **Label leakage.** 896 rows (10%) have titles that Amazon auto-filled ("Five Stars", "One Star"…), which spell out the label. They are removed from the model input.
- **Label noise.** Star ratings contradict the text in hand-checked examples, e.g. a 1★ "Perfect… Recommend" and a 5★ "Stopped working… would not buy again".
- **Duplicates.** 33 groups of identical inputs, 13 of them with conflicting labels ("ok" is both negative and neutral). The first copy of each is kept, so no text appears in both train and test.
- **Smaller issues.** 796 texts contain HTML `<br />` tags, 12 texts are empty, and one review of a Star Wars toy is filed under Cell Phones.

## Results (test set: 1,781 reviews, stratified 20%)

| Model | Macro-F1 | Accuracy | Neg. recall | Neutral recall | Pos. recall |
|---|---|---|---|---|---|
| **LogReg + BERT average** | **0.762** | **0.774** | **0.902** | 0.497 | 0.922 |
| TF-IDF (1–2-grams) + Logistic Regression | 0.746 | 0.745 | 0.757 | **0.655** | 0.822 |
| BERT `nlptown/bert-base-multilingual-uncased-sentiment` (pretrained, no fine-tuning) | 0.726 | 0.748 | 0.922 | 0.392 | 0.930 |
| VADER (lexicon baseline) | 0.467 | 0.525 | 0.567 | 0.105 | 0.905 |

Full per-class precision and recall are in `reports/results.csv`.
- **Headline:** averaging LR and BERT probabilities is best on macro-F1 and catches 90% of negative reviews, which is what triage needs.
- **Neutral is hardest:** 3★ reviews are hedged ("works, but…"). LR learns that hedging vocabulary from our data and recalls neutral best.
- **BERT caveat:** the nlptown model was fine-tuned on product reviews, so it may have seen Amazon-like text.

## Triage

- **Flag score:** `flag_score = 0.45·P(negative) + 0.35·issue severity + 0.10·helpful votes + 0.10·P(photo shows damage)`.
- **Weights:** the weights and severities in `triage.py` are **business assumptions**, because no ground-truth "problem order" label exists.
- **Photo check:** CLIP (`openai/clip-vit-base-patch32`) zero-shot scoring of 100 customer photos. It is **unevaluated** (no image labels). Mean P(damaged) is 0.63 for negative reviews' photos and 0.43 for positive ones.
- **Queue:** see notebook §5. Of the top 20, 19 are 1★. The 20th is a 5★ review that reports a dead product, which star-based routing would miss.

## Limitations

- Labels are star ratings, so label noise caps achievable accuracy.
- The test set is balanced, so neutral precision is optimistic against the live 7.3% neutral share.
- The keyword tagger has no sense of negation ("not cheaply made" still matches) and is unmeasured.
- The CLIP check cannot detect a *wrong* item without the listing photo.
- The flag-score weights are assumptions.

The next step is to join reviews to real returns and refunds, so the weights can be learned. See notebook §7 for the full table.

## Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
