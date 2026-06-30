# HELIOT Term Standardizer

HELIOT Term Standardizer is a prototype developed as an alternative to the previous LLM-based term extraction and standardization workflow used in HELIOT.

The goal of the project is to identify and standardize mentions of active ingredients, excipients, drug products, and drug brands in clinical notes. The system is based on a modular pipeline that combines deterministic dictionary matching with optional fallback strategies based on fuzzy matching and biomedical embeddings.

The pipeline follows a layered approach, moving from the most precise and efficient methods to more flexible fallback strategies:

```text
Exact matching
→ Fuzzy fallback
→ Semantic fallback
```

The main idea is to rely first on high-precision deterministic matching and to apply fallback strategies only to text spans that were not resolved by previous levels.

---

## Table of Contents

- [System Overview](#system-overview)
- [How to Install](#how-to-install)
- [How to Try](#how-to-try)
- [References](#references)

---

## System Overview

The system is built with a modular architecture composed of several independent components. The most important ones are:

### Knowledge Base Builder

The Knowledge Base Builder processes the input data and creates the structured resources used by the standardization pipeline.

It builds concepts and aliases for active ingredients, excipients, drug products, and drug brands. It also normalizes terms, assigns stable identifiers, and applies filtering rules to reduce noisy or unsafe aliases, such as very short acronyms, overly technical names, or ambiguous entries.

### Exact Matcher

The Exact Matcher is the main standardization layer. It uses the Aho-Corasick algorithm to index all safe aliases from the Knowledge Base and search them efficiently inside clinical notes.

Aho-Corasick allows thousands of terms to be searched at the same time with runtime approximately linear in the length of the input text. This avoids comparing each alias individually against the note and makes the exact matching layer very efficient.

### Fuzzy Fallback Matcher

The fuzzy fallback layer is used only on text spans that were not already resolved by exact matching.

It currently includes two strategies:

- **SymSpell**, useful for spelling errors, typos, and small edit-distance variations.
- **RapidFuzz with character n-gram blocking**, useful for multi-token variations, reordered tokens, and similar but non-identical expressions.

Both strategies apply confidence thresholds and ambiguity margins to reduce false positives.

### Semantic Fallback Matcher

The semantic fallback is an optional proof-of-concept layer based on NER, biomedical embeddings, and approximate nearest-neighbor search.

A pharmaceutical NER model is used to extract candidate drug mentions from the note. Each candidate is then encoded with a biomedical multilingual embedding model and compared against an ANN index built offline from Knowledge Base aliases.

### Runtime Caching

The system also includes in-memory runtime caching for expensive repeated operations, such as fuzzy lookup decisions and semantic embedding/search decisions.

---

## How to Install

To set up HELIOT Term Standardizer, follow these steps:

1. Install Python 3.11, if it is not already installed.
2. Clone the repository:

```bash
git clone https://github.com/FrancescoPinto02/heliot-term-standardizer.git
```

3. Navigate to the cloned repository directory:

```bash
cd /path/to/heliot-term-standardizer
```

4. Install Poetry:

```bash
pip install poetry
```

5. Create a new virtual environment with Python 3.11:

```bash
poetry env use python3.11
```

6. Activate the virtual environment:

```bash
poetry env activate
```

7. Install the project dependencies:

```bash
poetry install
```

---

## How to Try

To try HELIOT Term Standardizer, first complete the installation steps above. Then:

1. Configure the standardizer by editing:

```text
configs/default.yaml
```

2. Build the processed Knowledge Base:

```bash
poetry run python scripts/build_kb.py
```

3. Start the interactive CLI:

```bash
poetry run python scripts/interactive_cli.py
```

4. Insert clinical notes in the CLI to test the configured standardization pipeline.


> The `scripts` directory contains additional scripts that can be used to test specific components of the system.

---

## References

- HELIOT original repository: https://github.com/gadevito/heliot/tree/main
- HELIOT paper: https://arxiv.org/abs/2409.16395
