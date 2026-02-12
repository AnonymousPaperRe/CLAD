# CLAD

**CLAD: Complexity- and Logic-Aware Decomposition for Text-to-Cypher  
over Domain-Specific Knowledge Graphs**

CLAD is a research codebase for **logic-aware Text-to-Cypher generation** over domain-specific knowledge graphs.  
It explicitly models **query complexity, logical structure, and decomposition** to enable robust translation of complex natural language questions into executable Cypher queries.

This repository provides **datasets, training/testing code, and reasoning modules** for a multi-stage T5-based framework.

---

## ✨ Key Features

- Logic-aware decomposition for complex Text-to-Cypher queries
- Multi-stage pipeline with explicit intermediate representations
- Four task-specific **T5-base models**
- Support for multi-hop reasoning, conjunction, disjunction, and negation
- Modular design for reuse in other KG reasoning tasks

---

## 🧠 CLAD Framework

CLAD decomposes a natural language question into structured representations before Cypher generation.  
The pipeline consists of four task stages:

| Task | Description | Training Size |
|-----|-------------|---------------|
| **𝒯ₐ** | Atomic understanding (entities, relations, constraints) | 2,850 |
| **𝒯_dm** | Macro-level question decomposition | 2,750 |
| **𝒯_df** | Fine-grained / nested decomposition | 2,550 |
| **𝒯_f** | Logical formalization and Cypher construction | 2,510 |

Each task is trained with an **independent T5-base model**.

---

## 📁 Repository Structure

- `data/processed/` — task datasets for **𝒯ₐ**, **𝒯_dm**, **𝒯_df**, **𝒯_f** (train/dev/test).
- `models/` — checkpoints for the four T5-base task models (not tracked by git).
- `src/clad/pipeline/` — end-to-end multi-stage inference pipeline.
- `src/clad/reasoning/` — reusable reasoning utilities (e.g., `logic_reasoning.py`, grounding/alignment).
- `src/clad/training/` — training scripts (shared trainer + per-task launchers).
- `src/clad/evaluation/` — CM/EX evaluation, aggregation, and error taxonomy.
- `configs/` — YAML configs for data paths, schema settings, training hyperparameters, and evaluation.
- `outputs/` — predictions, logs, and generated tables.
