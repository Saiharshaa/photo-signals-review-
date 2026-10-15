# Walkthrough video script (~4 min, ~560 words)

Timestamps match the notebook sections. `video/record.py` scrolls the exported notebook on
these same timings, so if you change a timestamp here, change `SCENES` in `record.py` too.

---

**0:00–0:25 · Problem (title cell)**
A phone-accessories retailer gets thousands of reviews a day, as text and as customer photos. Reading them all isn't possible.
I want three things from each review: is it positive, neutral or negative; what went wrong; and should someone act on it today.
The data is about nine thousand reviews from the Cell Phones and Accessories category of Amazon Reviews 2023.

**0:25–1:10 · §1 Data audit**
The labels come from star ratings, so before trusting any metric I audited them.
In the wild only 7% of reviews are three-star, so I balanced to three thousand per class and report macro-F1.
Two problems stood out. First, leakage: when a customer leaves the title blank, Amazon fills in "Five Stars" or "One Star". That's 10% of rows, and the title literally is the label, so I strip it.
Second, label noise. Here's a one-star review that says "Perfect… recommend", and a five-star one that says "stopped working, would not buy again".

**1:10–1:55 · §2–3 Preprocessing and models**
After cleaning, 13 groups of identical texts had *different* labels. "Ok" was both negative and neutral, so I removed duplicates before an 80/20 stratified split.
I compared three models. VADER, a word list, gets macro-F1 0.47 and almost never says neutral.
TF-IDF with logistic regression, trained on our data, gets 0.75.
A pretrained BERT gets 0.73 without seeing our data. But it was tuned on product reviews, so some of that is familiarity.
Averaging the two gives the best result, 0.76, and catches 90% of negative reviews. That's what matters for triage.

**1:55–2:30 · §4 Explainability**
The logistic regression's weights make sense. Negative is "not", "doesn't", "waste", "junk". Positive is praise plus fit words like "fits" and "perfectly".
The interesting one is neutral: "but", "however", "otherwise", "not bad". A three-star review is a hedged review.
Of six errors I read, four were the model reading words, not situations. "Ridiculously cheap" meant low price here, not poor quality. Two were bad labels.

**2:30–3:15 · §5 Triage and images**
Next I combine signals. A keyword tagger finds the issue type: defect, counterfeit, wrong item, shipping damage or performance.
CLIP scores whether the customer photo shows damage. It's unevaluated, because there are no image labels, but the shattered cases rank at the top, and it rightly can't judge a wrong-colour complaint.
The flag score weights P(negative), issue severity, helpful votes and the photo. The weights are business assumptions, kept in one dictionary in `triage.py`.
Nineteen of the top twenty are one-star, and the twentieth is that five-star "stopped working" review. Routing by stars would have missed it.

**3:15–3:45 · §6 Live demo**
*(Switch to the Gradio tab.)* I'll paste a new review: "Received the wrong model and the box was crushed, charger is clearly a knockoff."
It comes out negative, tagged counterfeit, wrong item and shipping damage, with a high flag score.
A positive review scores near zero.

**3:45–4:10 · §7 Limitations**
To be honest about limits: star labels are noisy, the test set is balanced, the keyword tagger and image check are unmeasured, and the weights are assumptions.
The next step is joining reviews to real returns and refunds, so the weights can be learned instead of assumed. Thanks for watching.
