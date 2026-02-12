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

- `data/T5training` — datasets for **𝒯ₐ**, **𝒯_dm**, **𝒯_df**, **𝒯_f** (train).
- `data/testing`  — test datasets across 17 logic types (test).
- `clad/utils/` — source code for the clad pipeline.
- `clad/T5/` — source code for the T5 base training and inference.
- `clad/notebook/` — example code for question rewriting, complexity reasoning, logic reasoning, and Cypher synthesis.
- `baseline/` — baseline model prompts/outputs/scripts.
- `evaluation/` — metric CM and EX scripts.
