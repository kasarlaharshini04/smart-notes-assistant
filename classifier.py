"""Naive Bayes text classifier written from scratch (no ML library).

Idea: pick the category c that maximises
    P(c) * P(word1|c) * P(word2|c) * ...
We work with logs so many small probabilities don't underflow to 0.
"""
import math
import re
from collections import Counter

STOPWORDS = set("a an the is are on at to of in for and my be by it with this that "
                "i me we you your our".split())


def tokenize(text):
    """lowercase -> keep letters/digits -> drop stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class NaiveBayes:
    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        self.doc_count = Counter(labels)                 # notes per category
        self.word_count = {c: Counter() for c in self.classes}
        for text, label in zip(texts, labels):
            self.word_count[label].update(tokenize(text))
        self.total_words = {c: sum(self.word_count[c].values()) for c in self.classes}
        self.vocab = set()
        for c in self.classes:
            self.vocab |= set(self.word_count[c])
        self.n_docs = len(texts)
        return self

    def predict_proba(self, text):
        """Returns ({category: probability}, coverage).
        coverage = fraction of the note's words the model has seen in training."""
        # numbers/dates like "5", "28th" carry no topic info, so they don't count for coverage
        words = [w for w in tokenize(text) if not re.fullmatch(r"\d+(st|nd|rd|th)?", w)]
        known = [w for w in words if w in self.vocab]
        coverage = len(known) / len(words) if words else 0.0

        log_scores = {}
        for c in self.classes:
            score = math.log(self.doc_count[c] / self.n_docs)          # prior
            for w in known:                                            # unseen words are skipped
                # add-one (Laplace) smoothing so a word never seen in class c is not probability 0
                p = (self.word_count[c][w] + 1) / (self.total_words[c] + len(self.vocab))
                score += math.log(p)
            log_scores[c] = score

        # log-scores -> probabilities that sum to 1 (softmax, shifted by max for stability)
        m = max(log_scores.values())
        exp = {c: math.exp(s - m) for c, s in log_scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}, coverage

    def predict(self, text, threshold=0.0, min_coverage=0.0):
        """Returns (label, confidence, coverage). label is 'unsure' when the model
        is not confident enough OR hasn't seen enough of the note's words."""
        probs, coverage = self.predict_proba(text)
        best = max(probs, key=probs.get)
        conf = probs[best]
        if conf < threshold or coverage < min_coverage:
            return "unsure", conf, coverage
        return best, conf, coverage
