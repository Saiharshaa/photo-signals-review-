# Walkthrough narration (~4 min)

Each `##` block is one scene. Its header is the scene name, then `|`, then the notebook heading to show.
- `video/make_voice.py` reads this file and turns each block into speech. It writes `voice.m4a` and `timings.json`.
- `video/record.py` then scrolls each notebook section for exactly as long as its narration lasts.

Write numbers and acronyms the way they should be spoken, because the text is read aloud as-is.

## Problem | Review triage for a phone-accessories retailer
A phone accessories retailer gets thousands of reviews a day, as text and as customer photos, and nobody can read them all.
For each review I want to know three things. Is it positive, neutral or negative? What went wrong? And should someone act on it today?
The data is nine thousand reviews from the Cell Phones and Accessories category of Amazon Reviews 2023.

## Data audit | 1. Data audit
The labels come from star ratings, so before trusting any metric, I audited them.
In the wild, only seven percent of reviews are three star. So I balanced the sample to three thousand per class, and I report macro F1, which weights every class equally.
Two problems stood out. First, leakage. When a customer leaves the title blank, Amazon fills in "Five Stars" or "One Star". That is ten percent of the rows, and the title literally is the label, so I strip it.
Second, label noise. Here is a one star review that says "perfect, recommend", and a five star review that says "stopped working, would not buy again".

## Preprocessing and models | 2. Preprocessing
After cleaning, thirteen groups of identical texts had different labels. "Ok" was labelled both negative and neutral. So I removed duplicates before an eighty twenty stratified split.
Then I compared three models.
Vader, a sentiment word list, scores a macro F1 of 0.47, and almost never says neutral.
T F I D F with logistic regression, trained on our data, scores 0.75.
A pretrained Bert model scores 0.73 without seeing our data. But it was tuned on product reviews, so some of that is familiarity.
Averaging the two gives the best result, 0.76, and catches ninety percent of negative reviews. That is what matters for triage.

## Explainability | 4. Explainability
The logistic regression's weights make sense. Negative is "not", "doesn't", "waste" and "junk". Positive is praise, plus fit words like "fits" and "perfectly".
The interesting one is neutral: "but", "however", "otherwise", "not bad". A three star review is a hedged review.
Of six errors I read, four were the model reading words rather than situations. For example, "ridiculously cheap" meant a low price, not poor quality. The other two were wrong labels.

## Triage and images | 5. Combining signals
Next, I combine signals. A keyword tagger finds the type of issue: a defect, a counterfeit, a wrong item, shipping damage, or poor performance.
Then the Clip model scores whether the customer's photo shows damage. This part is unevaluated, because there are no image labels. But the shattered cases rank at the top, and it rightly cannot judge a wrong colour complaint.
The flag score weights the chance that a review is negative, the issue severity, helpful votes, and the photo. The weights are business assumptions, kept in one dictionary in the triage module.
Nineteen of the top twenty flagged reviews are one star. The twentieth is that five star "stopped working" review, which routing by stars would have missed.

## Live demo | 6. Live demo
Finally, the whole pipeline is wrapped in a Gradio app.
Given a new review that says "arrived with a cracked screen protector, and the charger stopped working after two days, looks fake", it predicts ninety three percent negative. It tags a defect, a counterfeit and shipping damage, and gives a flag score of 0.77.
The same function runs behind the public demo link.

## Limitations | 7. Limitations
To be clear about the limits: star labels are noisy, the test set is balanced, the keyword tagger and the image check are unmeasured, and the weights are assumptions.
The next step is joining reviews to real returns and refunds, so that the weights can be learned instead of assumed. Thanks for watching.
