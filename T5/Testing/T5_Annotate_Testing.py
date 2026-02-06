import json
import os
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---- paths ----
MODEL_DIR = ""
TEST_PATH = ""
RESULT_PATH = ""

MAX_GEN_LENGTH = 128
BATCH_SIZE = 8

logic_types = {
    1: "0p", 2: "1p", 3: "2p", 4: "2i",
    5: "2ni", 6: "2in", 7: "2nu", 8: "2u",
    9: "3p", 10: "3i", 11: "3u", 12: "pi",
    13: "ip", 14: "pu", 15: "iu", 16: "ui", 
    17: "pni", 18: "inp"
}

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_safely(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def annotate_file(model, tokenizer, device, in_path, out_path):
    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {in_path}, got: {type(data)}")

    # ensure "output"
    for inst in data:
        if "output" not in inst:
            inst["output"] = ""

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"Annotating {os.path.basename(in_path)}"):
        batch = data[i:i + BATCH_SIZE]
        inputs = [item.get("input", "") for item in batch]

        enc = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_length=MAX_GEN_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        preds = [p.strip() for p in preds]

        for item, pred in zip(batch, preds):
            item["output"] = pred

    write_json_safely(data, out_path)
    print(f"Saved: {out_path}  (samples: {len(data)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_id", type=int, required=True, help="1..19")
    args = parser.parse_args()

    if args.logic_id not in logic_types:
        raise ValueError(f"--logic_id must be in {sorted(logic_types.keys())}")

    lt = logic_types[args.logic_id]
    in_path = os.path.join(TEST_PATH, f"rewrite_{lt}.json")
    out_path = os.path.join(RESULT_PATH, f"annotated_{lt}.json")

    if not os.path.exists(in_path):
        print(f"[SKIP] Missing file: {in_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Logic type:", lt)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    os.makedirs(RESULT_PATH, exist_ok=True)
    annotate_file(model, tokenizer, device, in_path, out_path)

if __name__ == "__main__":
    main()
