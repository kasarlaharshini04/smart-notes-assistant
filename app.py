"""Smart Notes Assistant (command line). Run: python app.py
(run python make_data.py and python evaluate.py once first)"""
import csv
import json
import os

from classifier import NaiveBayes
from search import SearchIndex

NOTES_FILE = "my_notes.json"


def load_model():
    rows = list(csv.DictReader(open("data/notes.csv", encoding="utf-8")))
    rows += [r for r in csv.DictReader(open("data/handwritten.csv", encoding="utf-8"))
             if r["category"] != "unsure"]
    model = NaiveBayes().fit([r["text"] for r in rows], [r["category"] for r in rows])
    cfg = json.load(open("config.json")) if os.path.exists("config.json") else {"threshold": 0.45, "min_coverage": 0.2}
    return model, cfg


def classify(model, cfg, text):
    return model.predict(text, cfg["threshold"], cfg["min_coverage"])


def main():
    model, cfg = load_model()
    notes = json.load(open(NOTES_FILE)) if os.path.exists(NOTES_FILE) else []
    index = SearchIndex()
    for i, n in enumerate(notes):
        index.add(i, n["text"])

    def add(text):
        label, conf, _ = classify(model, cfg, text)
        notes.append({"text": text, "category": label})
        index.add(len(notes) - 1, text)
        json.dump(notes, open(NOTES_FILE, "w"), indent=1)
        return label, conf

    print("Smart Notes Assistant")
    while True:
        choice = input("\n1) Add note  2) Search  3) List notes  4) Load demo notes  5) Quit\n> ").strip()
        if choice == "1":
            text = input("Note: ").strip()
            if not text:
                continue
            label, conf = add(text)
            if label == "unsure":
                print(f"Not sure which category this is (confidence {conf:.0%}). Saved as 'unsure'.")
            else:
                print(f"Category: {label} (confidence {conf:.0%})")
        elif choice == "2":
            query = input("Search: ").strip()
            results = index.search(query, 5)
            if not results:
                print("No matching notes.")
            for i, score, text in results:
                print(f"  [{notes[i]['category']}] {text}   (score {score})")
        elif choice == "3":
            for i, n in enumerate(notes):
                print(f"  {i}. [{n['category']}] {n['text']}")
            if not notes:
                print("No notes yet.")
        elif choice == "4":
            for t in ["My Python exam is on Friday at 10 AM.", "Pay the electricity bill before the 28th.",
                      "Flight to Delhi is booked for Monday morning.", "Call grandma this weekend",
                      "Buy milk and bread from the store", "Dentist appointment on Tuesday at 4 PM"]:
                add(t)
            print("Loaded 6 demo notes (use option 3 to see how each was categorised).")
        elif choice == "5":
            break


if __name__ == "__main__":
    main()
