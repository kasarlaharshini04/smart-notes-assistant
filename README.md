# Smart Notes Assistant

A command-line app that saves everyday notes, **categorises them with a model trained from scratch**,
makes them **searchable**, and says **"unsure"** instead of forcing a note into the wrong category.
Pure Python 3, no external libraries.

## How to run
```
python make_data.py     # creates data/notes.csv and data/handwritten.csv
python evaluate.py      # trains, tunes the 'unsure' rule, prints all metrics, writes config.json
python app.py           # the app (option 4 loads demo notes)
```

## How it works
| Part | Technique | File |
|---|---|---|
| Categorising | Naive Bayes written from scratch: pick the category maximising P(category) x P(word given category) over the words, computed in logs, with add-one smoothing | `classifier.py` |
| "Unsure" | A note is "unsure" if the top probability is below a threshold **or** too few of its words were seen in training | `classifier.py`, tuned in `evaluate.py` |
| Search | Inverted index (hash map: word -> notes) with TF-IDF scoring; `heapq` returns the top 5 | `search.py` |
| Evaluation | Accuracy, confusion matrix, error list, Precision@5, MRR | `evaluate.py` |

Categories: study, bills, travel, appointments, shopping (+ "unsure").

## Data
No dataset was provided, so I made two:
- `data/notes.csv`: 183 **synthetic** notes generated from templates (16 topics, e.g. `python_exam`, `electricity_bill`).
- `data/handwritten.csv`: 42 notes I wrote by hand in different wording, including 12 that fit no category ("Call grandma this weekend").

Split: synthetic 80% train / 20% test. Hand-written notes are split three ways (train / dev / test, evenly per category).
The "unsure" settings are tuned **only on the hand-written dev notes**. The hand-written test notes are never used for tuning.

## Results (output of `python evaluate.py`)
| Measure | Result |
|---|---|
| Accuracy, synthetic test (37 notes) | 0.973 |
| Accuracy, hand-written in-scope test (10 notes), no unsure rule | 0.900 |
| Same, with the unsure rule | 0.800 |
| Out-of-scope notes correctly flagged "unsure" (4 notes) | 3/4 (without the rule: 0/4) |
| Search Precision@5 / MRR (17 queries) | 0.918 / 1.000 |

Chosen rule: confidence >= 0.45 and word coverage >= 0.2.

**Reading these numbers honestly:** the synthetic test is easy because the test notes come from the same templates as the
training notes. The hand-written results are the realistic ones, but the test sets are tiny (10 and 4 notes),
so one note changes the score a lot. Treat the numbers as indicative, not precise.

## What went wrong along the way
1. **First version of the unsure rule hurt valid notes.** Tuned on template notes, it looked good, but on hand-written test notes
   in-scope accuracy fell from 0.80 to 0.20 (7 of 10 valid notes wrongly flagged). Cause: the dev set was mostly easy template notes,
   and number tokens like "5th" were counted as unknown words. Fix: ignore numbers when measuring coverage, and tune on the hand-written dev
   notes only (see `evaluate.py`).
2. **Unseen words are the main weakness.** Naive Bayes skips words it never saw. "Pick up bananas, curd and onions" has no known words,
   so it looks identical to an out-of-scope note.

## Mistakes the model still makes
- `Clean the room and fold the clothes` (truth: unsure) -> predicted *appointments*, confidence 0.45, right on the threshold.
- `Bank visit to update the passbook on Monday at 10` -> "unsure" (confidence 0.34); "bank" and "passbook" never appeared in training.

## Limitations / next steps
- Small, partly synthetic data; more real notes would help most.
- Search matches exact words only ("test" will not find "exam"). Synonyms or embeddings would fix this.
- Search relevance is defined by topic labels, which are my own labels.
- Compare against scikit-learn's Naive Bayes and logistic regression as a sanity check.
