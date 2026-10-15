# CLAUDE.md — BC3415 Individual Assignment (Option A), FAST TRACK

## Goal
Ship a credible Option A submission in ~3–4 hours total: one notebook, one small module,
a Gradio demo, and a 3–5 min walkthrough video. Breadth over polish. Do not gold-plate.

Business case: retailer gets thousands of reviews daily (text + customer photos). Classify
sentiment (positive/neutral/negative), detect problems, combine signals to flag problematic orders.

Graded on: model performance, preprocessing pipeline, explainability/commentary,
creativity in combining text insights, reusability of code. Rubric rewards questioning data quality.

## Rules for you (Claude Code)
- Run all phases end to end without stopping, EXCEPT stop once after Phase 0 to confirm the dataset.
- Never invent metrics; only report numbers produced by code that ran.
- Plain, conventional code. I must be able to explain every line on camera.
- Each notebook section opens with a 2–3 line markdown cell: what, why, what to look at.
- Seed 42. Relative paths only. Notebook must pass Restart & Run All on CPU in under ~15 min.
- No fine-tuning (time budget). Inference with a pretrained transformer only.
- No Claude co-author / attribution trailers in commits (individual graded assignment).

## Phase 0 — Source the dataset (stop after this)
Search for and download a retail review dataset that has: review text, star rating, multiple
products, and ideally customer-uploaded image URLs.
First choice: McAuley-Lab "Amazon Reviews 2023" (HuggingFace), one mid-size category
(e.g. Electronics or Cell_Phones_and_Accessories), streamed or downloaded as jsonl.
If the HF loader fails, download the category jsonl.gz directly. Fallback: Kaggle
`mehmetisik/amazon-review` via kaggle CLI.
Build a sample of ~9,000 reviews: 3,000 per class (1–2★ neg, 3★ neutral, 4–5★ pos),
keeping the image-URL field. Save to data/sample.csv. Report: source, licence/terms, category,
rows per class, % of reviews with images, missing values. Then STOP and wait.

## Phase 1 — Audit + preprocessing (notebook §1–2)
Class counts in the raw category (natural imbalance) vs the balanced sample, and why we balanced.
Missing/empty text, duplicates, very short reviews. Show 5 reviews where stars contradict text
(label noise — labels come from stars, not text). Input = title + text. Stratified 80/20 split.

## Phase 2 — Models (notebook §3)
1. VADER baseline (no training).
2. TF-IDF (1–2 grams) + Logistic Regression (trained).
3. Pretrained `nlptown/bert-base-multilingual-uncased-sentiment` (predicts 1–5★; map to 3 classes),
   inference only on the test set. Note it may have seen Amazon-like reviews in pretraining.
One results table: macro-F1 (headline), per-class precision/recall, accuracy. Confusion matrix for
best two models. Save reports/results.csv.

## Phase 3 — Explain + combine (notebook §4–5)
- Top 15 n-grams per class from LR coefficients (bar chart).
- 6 misclassified examples, each tagged "label noise" or "model error".
- `triage.py`: keyword issue tagger (defect/dead-on-arrival, counterfeit, wrong item/misdelivery,
  damaged in shipping, performance, other) + flag score = weighted P(negative) + issue severity
  + helpful votes + image-damage probability. Weights in a dict at top of file, labelled as business
  assumptions (no ground truth exists). Output top-20 flagged review queue.
- Image check: CLIP (`openai/clip-vit-base-patch32`) zero-shot "damaged/broken/wrong product" vs
  "intact product" on up to 100 negative reviews that have images. Show 6 images with scores.
  State clearly it is unevaluated (no image labels).

## Phase 4 — Demo + README
Gradio cell in the notebook (share=True): review text (+ optional image URL) → sentiment probs,
issue tags, flag score. README: setup, how to run, results table, limitations, dataset citation.

## Phase 5 (optional) — Video assets
Write `video/script.md`: ~4 min narration with timestamps matched to notebook sections
(problem → data audit → models/results → explainability → triage + images → live demo → limitations).
Optionally `video/record.py`: export notebook to HTML, use Playwright (record_video) to scroll section
by section on the script's timings, then ffmpeg-merge with my voiceover file `video/voice.m4a`.
