import json
import glob
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "" Model path
TEST_DIR = "" Test dataset path
OUT_PATH = "" Output result path

MAX_INPUT_LENGTH = 512
MAX_GEN_LENGTH = 200
BATCH_SIZE = 16
NUM_BEAMS = 4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    examples = []  # (category, input, target)
    for path in sorted(glob.glob(os.path.join(TEST_DIR, "*.jsonl"))):
        name = os.path.basename(path)
        if name == "test.jsonl":
            continue
        category = name[:-len(".jsonl")]
        for line in open(path):
            r = json.loads(line)
            examples.append((category, r["input"], r["target"]))

    print(f"Loaded {len(examples)} test examples across "
          f"{len({c for c, _, _ in examples})} categories")

    results = []
    for i in range(0, len(examples), BATCH_SIZE):
        batch = examples[i:i + BATCH_SIZE]
        inputs = [b[1] for b in batch]

        enc = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_length=MAX_GEN_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )

        preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        for (category, inp, target), pred in zip(batch, preds):
            pred = pred.strip()
            results.append({
                "category": category,
                "input": inp,
                "target": target,
                "prediction": pred,
                "exact_match": pred == target.strip(),
            })

        print(f"[{i + len(batch)}/{len(examples)}] batch done")

    overall_n = len(results)
    overall_correct = sum(r["exact_match"] for r in results)

    per_category = {}
    for r in results:
        c = r["category"]
        stats = per_category.setdefault(c, {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += int(r["exact_match"])
    for c, stats in per_category.items():
        stats["exact_match"] = stats["correct"] / stats["n"]

    output = {
        "model": "t5-base-cypher",
        "model_dir": MODEL_DIR,
        "overall": {
            "n": overall_n,
            "correct": overall_correct,
            "exact_match": overall_correct / overall_n,
        },
        "per_category": per_category,
        "examples": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("==== Overall exact match:", output["overall"]["exact_match"], "====")
    print("Saved results to", OUT_PATH)


if __name__ == "__main__":
    main()
