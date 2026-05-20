from collections import Counter
from collections import defaultdict

def weighted_vote(predictions, weights):
    score = defaultdict(float)

    for model_name, pred in predictions.items():
        score[pred] += weights[model_name]

    return max(score, key=score.get)

def majority_vote(predictions):
    """Return the most common prediction (ties -> first encountered max)."""
    return Counter(predictions).most_common(1)[0][0]
