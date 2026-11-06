"""Run: python evaluate.py
Trains the model, tunes the 'unsure' rule on a dev set, reports results on data
the model never saw, evaluates search, and saves the chosen settings to config.json.
"""
import csv
import json
import random

from classifier import NaiveBayes
from search import SearchIndex


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def balanced_score(model, rows, threshold, min_cov):
    """average of: accuracy on in-scope notes and rate of out-of-scope notes flagged 'unsure'.
    (plain accuracy would let the many in-scope notes drown out the few out-of-scope ones)"""
    ins = [r for r in rows if r["category"] != "unsure"]
    outs = [r for r in rows if r["category"] == "unsure"]
    return (accuracy(model, ins, threshold, min_cov) + accuracy(model, outs, threshold, min_cov)) / 2


def accuracy(model, rows, threshold, min_cov):
    ok = 0
    for r in rows:
        label, _, _ = model.predict(r["text"], threshold, min_cov)
        ok += label == r["category"]
    return ok / len(rows)


def confusion(model, rows, threshold, min_cov):
    table = {}
    for r in rows:
        pred, _, _ = model.predict(r["text"], threshold, min_cov)
        table.setdefault(r["category"], {}).setdefault(pred, 0)
        table[r["category"]][pred] += 1
    return table


def print_confusion(table):
    cols = sorted({p for row in table.values() for p in row})
    print(f"{'true \\ predicted':<18}" + "".join(f"{c[:9]:>11}" for c in cols))
    for true in sorted(table):
        print(f"{true:<18}" + "".join(f"{table[true].get(c, 0):>11}" for c in cols))


def main():
    random.seed(1)
    notes = read_csv("data/notes.csv")
    hand = read_csv("data/handwritten.csv")
    random.shuffle(notes)

    # ---- split synthetic notes: 80% train / 20% test ----
    n = len(notes)
    train, test = notes[: int(.8 * n)], notes[int(.8 * n):]
    # handwritten notes (listed in groups of 6 per category) split 3 ways, evenly per category:
    #   every 3rd -> extra training data, next -> dev (tuning), last -> test (never touched)
    hand_train, hand_dev, hand_test = hand[0::3], hand[1::3], hand[2::3]
    train = train + [r for r in hand_train if r["category"] != "unsure"]  # 'unsure' is not a class to learn

    model = NaiveBayes().fit([r["text"] for r in train], [r["category"] for r in train])
    print(f"train={len(train)} test={len(test)} | handwritten train/dev/test={len(hand_train)}/{len(hand_dev)}/{len(hand_test)}\n")

    # ---- tune the 'unsure' rule on the handwritten dev notes (the realistic wording) ----
    dev = hand_dev
    best = (-1, 0.0, 0.0)
    for threshold in [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
        for min_cov in [0.0, 0.2, 0.3, 0.4, 0.5]:
            acc = balanced_score(model, dev, threshold, min_cov)
            if acc > best[0] + 1e-9:
                best = (acc, threshold, min_cov)
    _, threshold, min_cov = best
    print(f"[tuning] dev balanced score with no 'unsure' rule : {balanced_score(model, dev, 0, 0):.3f}")
    print(f"[tuning] best rule: confidence>={threshold}, coverage>={min_cov} -> dev balanced score {best[0]:.3f}\n")

    # ---- results on data never used for tuning ----
    print("=== CLASSIFIER RESULTS ===")
    print(f"Synthetic test set ({len(test)} notes)       : {accuracy(model, test, threshold, min_cov):.3f}")
    in_scope = [r for r in hand_test if r["category"] != "unsure"]
    out_scope = [r for r in hand_test if r["category"] == "unsure"]
    print(f"Handwritten in-scope, no unsure rule ({len(in_scope)})   : {accuracy(model, in_scope, 0, 0):.3f}")
    print(f"Handwritten in-scope, with unsure rule     : {accuracy(model, in_scope, threshold, min_cov):.3f}")
    flagged = sum(model.predict(r["text"], threshold, min_cov)[0] == "unsure" for r in out_scope)
    print(f"Out-of-scope notes correctly flagged unsure: {flagged}/{len(out_scope)}")
    wrongly = sum(model.predict(r["text"], threshold, min_cov)[0] == "unsure" for r in in_scope)
    print(f"In-scope notes wrongly flagged unsure      : {wrongly}/{len(in_scope)}")
    without = sum(model.predict(r["text"], 0, 0)[0] != "unsure" for r in out_scope)
    print(f"(without the rule, {without}/{len(out_scope)} out-of-scope notes get forced into a category)\n")

    print("Confusion matrix on handwritten test notes (rows = truth):")
    print_confusion(confusion(model, hand_test, threshold, min_cov))

    print("\nMistakes on handwritten test notes:")
    for r in hand_test:
        pred, conf, cov = model.predict(r["text"], threshold, min_cov)
        if pred != r["category"]:
            print(f"  '{r['text']}'\n     truth={r['category']} predicted={pred} (conf={conf:.2f}, coverage={cov:.2f})")

    # ---- search evaluation ----
    print("\n=== SEARCH RESULTS ===")
    index = SearchIndex()
    topic_of = {}
    for r in notes:
        index.add(int(r["id"]), r["text"])
        topic_of[int(r["id"])] = r["topic"]
    queries = [
        ("python exam", "python_exam"), ("when is my python test", "python_exam"),
        ("dsa assignment", "dsa_assignment"), ("maths test", "maths_exam"),
        ("electricity bill", "electricity_bill"), ("rent payment", "rent"),
        ("credit card due", "credit_card"), ("internet wifi bill", "internet_bill"),
        ("flight booking", "flight"), ("train tickets", "train"),
        ("hotel reservation", "hotel"), ("dentist appointment", "dentist"),
        ("client meeting", "client_meeting"), ("interview schedule", "interview"),
        ("grocery list", "groceries"), ("buy headphones", "gadgets"),
        ("buy shoes shirts", "clothes"),
    ]
    k = 5
    p_at_k, rr = [], []
    for q, topic in queries:
        results = index.search(q, k)
        hits = [topic_of[i] == topic for i, _, _ in results]
        p_at_k.append(sum(hits) / k)
        rr.append(next((1 / (pos + 1) for pos, h in enumerate(hits) if h), 0.0))
        if not hits or not hits[0]:
            print(f"  weak query: '{q}' -> top result topic: {topic_of[results[0][0]] if results else 'none'}")
    print(f"{len(queries)} labelled queries (a result is relevant if its topic matches)")
    print(f"Precision@{k}: {sum(p_at_k) / len(p_at_k):.3f}   MRR: {sum(rr) / len(rr):.3f}")

    with open("config.json", "w") as f:
        json.dump({"threshold": threshold, "min_coverage": min_cov}, f)
    print("\nSaved settings to config.json")


if __name__ == "__main__":
    main()
