import json
import os
import re
import argparse
from typing import Tuple, Optional, Dict, Any, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# Config
# -----------------------
MODEL_DIR_MEDIUM = ""
MODEL_DIR_FINAL  = ""

IN_DIR  = ""
OUT_DIR = ""

MAX_GEN_LENGTH = 256
BATCH_SIZE = 8

# Per-side budgets (NOT including the top-level call)
MAX_CALLS_S1 = 3
MAX_CALLS_S2 = 3

logic_types: Dict[int, str] = {
    1: "0p", 2: "1p", 3: "2p", 4: "2i",
    5: "2ni", 6: "2in", 7: "2nu", 8: "2u",
    9: "3p", 10: "3i", 11: "3u", 12: "pi",
    13: "ip", 14: "pu", 15: "up", 16: "pni",
    17: "inp"
}

# -----------------------
# IO helpers
# -----------------------
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_safely(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# Text helpers
# -----------------------
def fix_node_spacing(s: str) -> str:
    s = re.sub(r'(?<!\s)(\[(?:node_\d+)\])', r' \1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def cleanup_t5_specials(s: str) -> str:
    s = re.sub(r'</s>|<pad>', '', s)
    return s.strip()

# def parse_decompose_output(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
#     """
#     Parse: S1:... | S2:... | logic:OR
#     Return (s1, s2, logic)
#     """
#     if not s or not isinstance(s, str):
#         return None, None, None

#     parts = [p.strip() for p in s.split("|")]
#     s1 = s2 = logic = None

#     for p in parts:
#         pl = p.lower()
#         if pl.startswith("s1:"):
#             s1 = p[3:].strip()
#         elif pl.startswith("s2:"):
#             s2 = p[3:].strip()
#         elif pl.startswith("logic:"):
#             logic = p[6:].strip().upper()

#     return s1, s2, logic

# def parse_decompose_output(s: str):
#     if not s or not isinstance(s, str):
#         return None, None, None

#     m = re.search(
#         r"S1\s*:\s*(.*?)\s*\|\s*S2\s*:\s*(.*?)\s*\|\s*logic\s*:\s*([A-Za-z]+)",
#         s, flags=re.IGNORECASE | re.DOTALL
#     )
#     if not m:
#         return None, None, None

#     s1 = m.group(1).strip()
#     s2 = m.group(2).strip()
#     logic = m.group(3).strip().upper()
#     return s1, s2, logic

def parse_decompose_output(s: str):
    if not s or not isinstance(s, str):
        return None, None, None

    # 1) Preferred strict format
    m = re.search(
        r"S1\s*:\s*(.*?)\s*\|\s*S2\s*:\s*(.*?)\s*\|\s*logic\s*:\s*([A-Za-z]+)",
        s, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip().upper()

    # 2) Fallback: "<clause1> | <clause2> | logic:AND"
    m = re.search(
        r"^(.*?)\s*\|\s*(.*?)\s*\|\s*logic\s*:\s*([A-Za-z]+)\s*$",
        s, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip().upper()

    return None, None, None

def get_node_level(text: str):
    ids = re.findall(r'\[node_(\d+)\]', text or "")
    unique_ids = sorted(set(ids), key=int)
    return len(unique_ids), [f"[node_{i}]" for i in unique_ids]

# -----------------------
# Model loading / routing
# -----------------------
def load_model_and_tokenizer(model_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    mdl.eval()
    return tok, mdl

def choose_model_dir_by_level(level: int) -> str:
    # your rule:
    # level > 3 -> MEDIUM
    # level == 3 -> FINAL
    # level < 3 -> (won't decompose), default FINAL
    if level > 3:
        return MODEL_DIR_MEDIUM
    return MODEL_DIR_FINAL

# -----------------------
# Decomposition helpers
# -----------------------
def decompose_text_once(models: Dict[str, Any], device: torch.device, text: str) -> Dict[str, Any]:
    level, _ = get_node_level(text)
    model_dir = choose_model_dir_by_level(level)
    tok = models[model_dir]["tokenizer"]
    mdl = models[model_dir]["model"]

    enc = tok([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = mdl.generate(**enc, max_length=MAX_GEN_LENGTH, num_beams=4, early_stopping=False)

    pred = tok.decode(out_ids[0], skip_special_tokens=False).strip()
    pred = cleanup_t5_specials(pred)
    pred = fix_node_spacing(pred)

    s1, s2, logic = parse_decompose_output(pred)
    return {"raw": pred, "s1": s1, "s2": s2, "logic": logic, "level_in": level}

def build_side_tree(models: Dict[str, Any], device: torch.device, text: str, max_calls: int,
                    side_name: str, events: List[dict]):
    calls = 0

    def _rec(curr_text: str, path: str):
        nonlocal calls
        level, _ = get_node_level(curr_text)

        if level < 3:
            return {"type": "leaf", "text": curr_text, "level": level}

        if calls >= max_calls:
            return {"type": "leaf", "text": curr_text, "level": level, "stopped": True, "reason": "call_limit"}

        calls += 1
        res = decompose_text_once(models, device, curr_text)

        # parse fail => stop
        if not res["s1"] or not res["s2"] or not res["logic"]:
            events.append({
                "side": side_name,
                "call": calls,
                "path": path,
                "input": curr_text,
                "raw": res["raw"],
                "level_in": res["level_in"],
                "stopped": True,
                "reason": "parse_fail",
            })
            return {
                "type": "leaf",
                "text": curr_text,
                "level": level,
                "stopped": True,
                "reason": "parse_fail",
                "raw": res["raw"],
            }

        lvl_s1 = get_node_level(res["s1"])[0]
        lvl_s2 = get_node_level(res["s2"])[0]
        terminal = (lvl_s1 < 3) and (lvl_s2 < 3)

        events.append({
            "side": side_name,
            "call": calls,
            "path": path,
            "input": curr_text,
            "raw": res["raw"],
            "logic": res["logic"],
            "s1": res["s1"],
            "s2": res["s2"],
            "level_in": res["level_in"],
            "level_s1": lvl_s1,
            "level_s2": lvl_s2,
            "stopped": terminal,
            "reason": "children_level<3" if terminal else None,
        })

        left = _rec(res["s1"], path + ".s1")
        right = _rec(res["s2"], path + ".s2")

        return {
            "type": "branch",
            "input": curr_text,
            "raw": res["raw"],
            "logic": res["logic"],
            "level": level,
            "s1": left,
            "s2": right
        }

    if not text or not text.strip():
        return {"type": "leaf", "text": text, "level": 0, "stopped": True, "reason": "empty_input"}, 0

    tree = _rec(text, side_name)
    return tree, calls

def process_one_file(models: Dict[str, Any], device: torch.device, lt: str, in_path: str, out_path: str):
    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {in_path}, got: {type(data)}")

    # init fields
    for inst in data:
        annotated = inst.get("annotated", "")
        level, _ = get_node_level(annotated)
        inst["level"] = level
        inst.setdefault("decomposed", None)

    to_run = [inst for inst in data if inst.get("level", 0) >= 3 and inst.get("annotated", "").strip()]
    print(f"\n[{lt}] Total: {len(data)} | To process (level>=3): {len(to_run)}")

    for inst in tqdm(to_run, desc=f"Decomposing ({lt})"):
        annotated = inst.get("annotated", "")
        events: List[dict] = []

        top_level_in = inst.get("level", get_node_level(annotated)[0])
        top = decompose_text_once(models, device, annotated)

        lvl_s1 = get_node_level(top["s1"])[0] if top["s1"] else None
        lvl_s2 = get_node_level(top["s2"])[0] if top["s2"] else None

        top_struct = {
            "input": annotated,
            "raw": top["raw"],
            "logic": top["logic"],
            "s1": top["s1"],
            "s2": top["s2"],
            "level_in": top_level_in,
            "level_s1": lvl_s1,
            "level_s2": lvl_s2,
        }

        if not top["s1"] or not top["s2"] or not top["logic"]:
            inst["decomposed"] = {
                "top": top_struct,
                "sides": {
                    "s1": {"tree": None, "calls": 0, "max_calls": MAX_CALLS_S1},
                    "s2": {"tree": None, "calls": 0, "max_calls": MAX_CALLS_S2},
                },
                "events": events,
                "note": "top_level_parse_fail"
            }
            continue

        tree_s1, calls_s1 = build_side_tree(models, device, top["s1"], MAX_CALLS_S1, "s1", events)
        tree_s2, calls_s2 = build_side_tree(models, device, top["s2"], MAX_CALLS_S2, "s2", events)

        inst["decomposed"] = {
            "top": top_struct,
            "sides": {
                "s1": {"tree": tree_s1, "calls": calls_s1, "max_calls": MAX_CALLS_S1},
                "s2": {"tree": tree_s2, "calls": calls_s2, "max_calls": MAX_CALLS_S2},
            },
            "events": events
        }

    write_json_safely(data, out_path)
    print(f"[{lt}] Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_id", type=int, required=True, help="1..19")
    args = parser.parse_args()

    if args.logic_id not in logic_types:
        raise ValueError(f"--logic_id must be in {sorted(logic_types.keys())}")

    lt = logic_types[args.logic_id]
    in_path = os.path.join(IN_DIR, f"annotated_{lt}.json")
    out_path = os.path.join(OUT_DIR, f"decomposed_{lt}.json")

    if not os.path.exists(in_path):
        print(f"[SKIP] Missing file: {in_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "| logic type:", lt)

    # load both models once per array task
    models: Dict[str, Any] = {}
    for md in [MODEL_DIR_MEDIUM, MODEL_DIR_FINAL]:
        tok, mdl = load_model_and_tokenizer(md, device)
        models[md] = {"tokenizer": tok, "model": mdl}
    print("Loaded models OK.")

    process_one_file(models, device, lt, in_path, out_path)

if __name__ == "__main__":
    main()
