"""Review triage helpers: text cleaning, keyword issue tagging and a flag score.

Every number in WEIGHTS, SEVERITY and HELPFUL_CAP is a BUSINESS ASSUMPTION, not a learned
value. The data has no ground-truth "problematic order" label, so these encode how a
retailer might prioritise. Change them here and the notebook and the demo pick them up.
"""
import html
import math
import re

import pandas as pd

# How much each signal contributes to the flag score. They sum to 1, so the score is 0-1.
WEIGHTS = {
    "p_negative": 0.45,      # sentiment model's probability that the review is negative
    "issue_severity": 0.35,  # worst issue found by the keyword tagger
    "helpful_votes": 0.10,   # other shoppers marked the review as helpful
    "image_damage": 0.10,    # CLIP probability that the customer photo shows a damaged/wrong product
}

# How serious each issue is for the retailer (0-1).
# Assumption: counterfeit (legal/brand risk) > dead on arrival > shipping damage > wrong item
# > performance complaints. "other" means no specific problem keyword was found.
SEVERITY = {
    "counterfeit": 1.0,
    "defect_doa": 0.9,
    "damaged_in_shipping": 0.8,
    "wrong_item": 0.7,
    "performance": 0.5,
    "other": 0.0,
}

HELPFUL_CAP = 20  # helpful votes at which the helpful signal reaches its maximum of 1

# Keyword patterns per issue (regular expressions, matched on lower-cased text).
ISSUE_KEYWORDS = {
    "defect_doa": [
        r"dead on arrival", r"\bdoa\b", r"stopped working", r"quit working", r"not working",
        r"(doesn't|does not|didn't|did not|won't|will not|wouldn't) (work|charge|turn on|power on|connect|pair)",
        r"defective", r"faulty", r"\bdied\b", r"malfunction", r"\bbroke (after|within)",
    ],
    "counterfeit": [
        r"\bfake\b", r"counterfeit", r"knock[- ]?off", r"\breplica\b", r"\bbootleg\b",
        r"not (genuine|authentic|original|oem|real)",
    ],
    "wrong_item": [
        r"wrong (item|color|colour|size|model|product|one|case|cable|charger)",
        r"(received|sent|got) (me )?(the |a )?wrong", r"not what i (ordered|paid for)",
        r"not as (pictured|described|shown|advertised)", r"never (arrived|received|came)",
        r"missing (parts?|pieces?|items?|screws?)",
    ],
    "damaged_in_shipping": [
        r"(arrived|came|showed up|delivered) (already )?(broken|damaged|cracked|scratched|shattered|bent)",
        r"(box|package|packaging) (was |is )?(broken|damaged|crushed|torn|open)",
        r"damaged in (shipping|transit|the mail)", r"\bdamaged\b", r"\bcracked\b", r"\bshattered\b",
    ],
    "performance": [
        r"(charges?|charging) (very |so |really )?slow", r"slow(ly)? charg", r"overheat",
        r"gets? (very |really |so )?hot", r"disconnect", r"battery (drains?|dies)",
        r"(weak|poor|bad) (signal|reception|connection|sound|quality)", r"flimsy", r"cheaply made",
        r"(doesn't|does not|didn't|did not) fit", r"falls? off", r"peel(s|ed|ing)? off", r"\bbubbles?\b",
    ],
}

# Amazon fills in "Five Stars", "One Star", ... when a customer leaves the title blank.
# Those titles contain the answer (the star rating), so they must not be used as model input.
AUTO_TITLE_PATTERN = r"^(one|two|three|four|five) stars?$"


def star_to_label(rating):
    """1-2 stars = negative, 3 = neutral, 4-5 = positive."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def clean_text(text):
    """Decode HTML entities, drop HTML tags, use plain apostrophes, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def make_input(title, text):
    """Model input = title + text, skipping auto-generated 'N Stars' titles."""
    title = clean_text(title)
    if re.match(AUTO_TITLE_PATTERN, title, flags=re.IGNORECASE):
        title = ""
    text = clean_text(text)
    if title and text:
        return f"{title}. {text}"
    return title or text


def tag_issues(text):
    """Return the list of issue types whose keywords appear in the text (or ['other'])."""
    text = clean_text(text).lower()
    tags = []
    for issue, patterns in ISSUE_KEYWORDS.items():
        if any(re.search(p, text) for p in patterns):
            tags.append(issue)
    return tags or ["other"]


def issue_severity(tags):
    """Severity of the worst issue in the list."""
    return max(SEVERITY[t] for t in tags)


def helpful_score(votes):
    """Scale helpful votes to 0-1 on a log scale, so 1 vote matters but 200 is not 10x 20."""
    if votes is None or pd.isna(votes) or votes <= 0:
        return 0.0
    return min(math.log1p(votes) / math.log1p(HELPFUL_CAP), 1.0)


def flag_score(p_negative, tags, helpful_votes=0, image_damage=None):
    """Weighted sum of the four signals. A missing photo contributes 0 (no extra evidence)."""
    if image_damage is None or pd.isna(image_damage):
        image_damage = 0.0
    return (WEIGHTS["p_negative"] * p_negative
            + WEIGHTS["issue_severity"] * issue_severity(tags)
            + WEIGHTS["helpful_votes"] * helpful_score(helpful_votes)
            + WEIGHTS["image_damage"] * image_damage)


def build_queue(reviews, top_n=20):
    """Score every review and return the top_n most urgent ones.

    `reviews` needs columns: input, p_negative, helpful_vote, image_damage.
    """
    out = reviews.copy()
    out["issues"] = out["input"].apply(tag_issues)
    out["flag_score"] = [
        flag_score(p, tags, votes, img)
        for p, tags, votes, img in zip(out["p_negative"], out["issues"], out["helpful_vote"], out["image_damage"])
    ]
    out["issues"] = out["issues"].apply(", ".join)
    return out.sort_values("flag_score", ascending=False).head(top_n)
