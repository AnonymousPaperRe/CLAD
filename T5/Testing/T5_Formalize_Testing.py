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
    13: "ip", 14: "pu", 15: "iu",
    16: "ui", 17: "pni", 18: "inp"
}

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_safely(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# Tree utilities
# -----------------------
def iter_leaf_nodes(tree):
    """
    Yield leaf dicts (mutable references) from a tree.
    Leaf: {"type":"leaf","text": "...", ...}
    Branch: {"type":"branch","s1": <subtree>, "s2": <subtree>, ...}
    """
    if tree is None or not isinstance(tree, dict):
        return
    t = tree.get("type")
    if t == "leaf":
        yield tree
    elif t == "branch":
        yield from iter_leaf_nodes(tree.get("s1"))
        yield from iter_leaf_nodes(tree.get("s2"))

def collect_leaf_text_jobs(inst):
    """
    For non-0p/1p instances:
    collect all leaf nodes under decomposed.sides.s1.tree and decomposed.sides.s2.tree,
    return a list of (leaf_node_dict, input_text).
    """
    jobs = []
    dec = inst.get("decomposed") or {}
    sides = dec.get("sides") or {}

    for side_name in ("s1", "s2"):
        side = sides.get(side_name) or {}
        tree = side.get("tree")
        for leaf in iter_leaf_nodes(tree):
            txt = (leaf.get("text") or "").strip()
            if not txt:
                continue
            jobs.append((leaf, txt))
    return jobs

# -----------------------
# Model inference
# -----------------------
def run_generate(model, tokenizer, device, inputs):
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
    return [p.strip() for p in preds]

# -----------------------
# Main processing
# -----------------------
def formalize_file(model, tokenizer, device, in_path, out_path, lt):
    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {in_path}, got: {type(data)}")

    # ---- Case A: 0p / 1p => formalize inst["annotated"] into inst["formalized"]
    if lt in ("0p", "1p"):
        # prepare
        for inst in data:
            inst.setdefault("formalized", None)

        # build batch inputs
        jobs = []
        for inst in data:
            annotated = (inst.get("annotated") or "").strip()
            if not annotated:
                inst["formalized"] = None
                continue
            jobs.append(inst)

        for i in tqdm(range(0, len(jobs), BATCH_SIZE), desc=f"Formalizing {lt} (annotated)"):
            batch = jobs[i:i + BATCH_SIZE]
            inputs = [(b.get("annotated") or "").strip() for b in batch]
            preds = run_generate(model, tokenizer, device, inputs)

            for inst, pred in zip(batch, preds):
                inst["formalized"] = pred.replace("<pad> ", "").replace("</s>", "")

        write_json_safely(data, out_path)
        print(f"Saved: {out_path}  (samples: {len(data)})")
        return

    # ---- Case B: others => formalize each leaf["text"] and write into leaf["formalized"]
    # ensure leaf["formalized"] exists (optional)
    for inst in data:
        dec = inst.get("decomposed") or {}
        sides = dec.get("sides") or {}
        for side_name in ("s1", "s2"):
            side = sides.get(side_name) or {}
            tree = side.get("tree")
            for leaf in iter_leaf_nodes(tree):
                leaf.setdefault("formalized", None)

    # collect all leaf jobs across dataset
    leaf_jobs = []  # list of (leaf_dict_ref, input_text)
    for inst in data:
        leaf_jobs.extend(collect_leaf_text_jobs(inst))

    if not leaf_jobs:
        write_json_safely(data, out_path)
        print(f"Saved: {out_path}  (no leaf jobs)")
        return

    # run in batches
    for i in tqdm(range(0, len(leaf_jobs), BATCH_SIZE), desc=f"Formalizing {lt} (leaf texts)"):
        batch = leaf_jobs[i:i + BATCH_SIZE]
        inputs = [txt for (_, txt) in batch]
        preds = run_generate(model, tokenizer, device, inputs)

        for (leaf, _), pred in zip(batch, preds):
            leaf["formalized"] = pred.replace("<pad> ", "").replace("</s>", "")

    write_json_safely(data, out_path)
    print(f"Saved: {out_path}  (samples: {len(data)} | leaves: {len(leaf_jobs)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_id", type=int, required=True, help="1..19")
    args = parser.parse_args()

    if args.logic_id not in logic_types:
        raise ValueError(f"--logic_id must be in {sorted(logic_types.keys())}")

    lt = logic_types[args.logic_id]
    in_path = os.path.join(TEST_PATH, f"decomposed_{lt}.json")
    out_path = os.path.join(RESULT_PATH, f"formalized_{lt}.json")

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
    formalize_file(model, tokenizer, device, in_path, out_path, lt)

if __name__ == "__main__":
    main()
