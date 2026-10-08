"""Score one person's AQ-10 answers with the trained model.

This is a screening aid, not a diagnosis. A high score means a full
assessment by a qualified clinician is worth arranging; a low score does not
rule autism out.

Usage (all inputs as flags):
    python predict.py --scores 1,0,1,1,0,1,1,0,1,1 --age 24 --gender f \
        --jaundice no --family-history no --used-app-before no --relation Self

Usage (interactive - asks for anything not given as a flag):
    python predict.py
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from autism_model import AQ_CUTOFF, AQ_ITEM_TOPICS, AQ_ITEMS

YES_NO = {"y": "yes", "yes": "yes", "n": "no", "no": "no"}


def ask(prompt: str, valid=None, cast=str):
    while True:
        raw = input(prompt).strip()
        if valid is not None:
            key = raw.lower()
            if key in valid:
                return valid[key] if isinstance(valid, dict) else key
            print(f"  Please enter one of: {', '.join(sorted(set(valid)))}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  Invalid value, try again.")


def ask_scores() -> list[int]:
    print(
        "\nAnswer each AQ-10 item using the official AQ-10 Adult sheet, then enter\n"
        "its score: 1 if the answer was on the side shown in [brackets]\n"
        "(either 'definitely' or 'slightly'), otherwise 0.\n"
    )
    scores = []
    for i, item in enumerate(AQ_ITEMS, 1):
        topic, direction = AQ_ITEM_TOPICS[item]
        scores.append(int(ask(f"  Q{i:>2}. {topic} [{direction}] -> score (0/1): ", valid={"0": "0", "1": "1"})))
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="models/autism_model.joblib")
    p.add_argument("--scores", help="ten comma-separated 0/1 values for A1..A10")
    p.add_argument("--age", type=float)
    p.add_argument("--gender", choices=["m", "f"])
    p.add_argument("--jaundice", choices=["yes", "no"], help="had jaundice at birth")
    p.add_argument("--family-history", choices=["yes", "no"], help="immediate family member diagnosed with autism")
    p.add_argument("--used-app-before", choices=["yes", "no"], help="used a screening app before")
    p.add_argument("--relation", help="who is filling this in: Self, Parent, Relative, Health care professional, Others")
    p.add_argument("--ethnicity", help="only needed if the model was trained with --with-demographics")
    p.add_argument("--country", help="only needed if the model was trained with --with-demographics")
    args = p.parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]

    if args.scores:
        scores = [int(s) for s in args.scores.split(",")]
        if len(scores) != 10 or any(s not in (0, 1) for s in scores):
            p.error("--scores needs exactly ten values, each 0 or 1")
    else:
        scores = ask_scores()

    row = dict(zip(AQ_ITEMS, scores))
    row["age"] = args.age if args.age is not None else ask("Age in years: ", cast=float)
    row["gender"] = args.gender or ask("Gender (m/f): ", valid={"m": "m", "f": "f"})
    row["jaundice"] = args.jaundice or ask("Jaundice at birth? (yes/no): ", valid=YES_NO)
    row["austim"] = args.family_history or ask("Immediate family member with autism? (yes/no): ", valid=YES_NO)
    row["used_app_before"] = args.used_app_before or ask("Used a screening app before? (yes/no): ", valid=YES_NO)
    row["relation"] = args.relation or ask("Who is answering (Self/Parent/Relative/Others): ") or "Self"
    if bundle["with_demographics"]:
        row["ethnicity"] = args.ethnicity or ask("Ethnicity: ")
        row["contry_of_res"] = args.country or ask("Country of residence: ")

    X = pd.DataFrame([row])[bundle["features"]]
    total = sum(scores)
    proba = model.predict_proba(X)[0, 1]

    print("\n" + "=" * 56)
    print(f"AQ-10 total:            {total}/10  (referral cut-off is {AQ_CUTOFF})")
    print(f"Model ({bundle['model_name']}): {proba:.0%} estimated likelihood")
    flagged = total >= AQ_CUTOFF or proba >= 0.5
    if flagged:
        print("Result: consider a full assessment by a qualified clinician.")
    else:
        print("Result: below the screening threshold.")
    print("-" * 56)
    print("This is a screening aid, not a diagnosis. Only a clinician can\n"
          "diagnose autism. A low score does not rule it out.")
    print("=" * 56)


if __name__ == "__main__":
    main()
