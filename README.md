# Sourashtra Language NMT — Progressive Neural Machine Translation

A research project building a **Sourashtra-to-English/Tamil neural machine translation** system for an endangered Indo-Aryan language spoken by ~500,000 people in southern India.

Five model generations, a hybrid retrieval system, a web application, and an IEEE conference paper — all built on a cleaned dataset of 12,771 dictionary entries.

---

## Results Summary

| Model | Architecture | Exact Match | BLEU |
|-------|-------------|-------------|------|
| V1 | Character-level GRU + Bahdanau Attention | 0.42% | — |
| V2 | Transformer + BPE Subword Tokenization | ~3% | — |
| V3 | T5-small Fine-tuned | 6.01% | — |
| V4 | T5-small + Tamil Cross-lingual Transfer | 5.80% | — |
| V5 | ByT5-small (Byte-level) | 9.25% | 4.70 |
| **V5 Hybrid** | **ByT5 + Levenshtein/Jaccard Retrieval** | **9.61%** | **4.70** |

V5 Hybrid represents a **22.9× improvement** over the V1 baseline.

---

## Project Structure

```
.
├── app.py                        # Flask web application (real-time translation UI)
├── hybrid_translate.py           # Hybrid neural + retrieval inference engine
├── inference.py                  # Standalone inference script
│
├── config.py / config_v[2-5].py  # Hyperparameters for each model generation
├── model.py / model_v2.py        # Model architecture definitions
├── data_loader*.py               # Data pipelines (character, BPE, HuggingFace)
├── train*.py                     # Training scripts for each version
├── evaluate*.py                  # Evaluation scripts (BLEU, chrF, exact-match)
├── metrics.py                    # Shared evaluation metric utilities
├── generate_paper_figures.py     # Figure generation for the IEEE paper
├── capture_screenshots.py        # Screenshot utility for the web app
│
├── Dataset/                      # Raw source data (CSV/XLS files)
├── cleaned_data/                 # Unified cleaned dataset (12,771 entries)
│   ├── unified_full_dataset.csv  # Main dataset
│   ├── translation_roman_english.csv
│   ├── example_sentences.csv
│   └── by_category/              # 106 semantic category files
│
├── tokenizers/                   # Trained BPE SentencePiece models (V2)
├── results*/                     # Evaluation results JSON per version
├── paper_figures/                # Matplotlib figures for the IEEE paper
│
├── website/                      # Static web application
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── ieee_paper/                   # IEEE conference paper (LaTeX + PDF)
│   ├── sourashtra_nmt_paper*.tex
│   └── *.pdf
│
├── sourashtra-dictionary-main/   # Original dictionary source data
└── claude_files/                 # Extended project documentation
    ├── README.md
    └── PROJECT_REPORT.md
```

> **Not tracked:** `.venv/`, `checkpoints*/`, `logs*/`, `*.pt`, `*.safetensors`  
> Model weights are large (39 MB – 1.2 GB); store on HuggingFace Hub if sharing.

---

## Dataset

- **12,771** cleaned, deduplicated dictionary entries (down from 40,774 raw)
- **521** duplicate entries removed
- **106** semantic categories
- **Multi-script**: Sourashtra script, Tamil, Devanagari, Romanized, IPA, IAST
- **2,346** parallel example sentences (Sourashtra / English / Tamil)

Data sources: `sourashtra-dictionary-main` (words + corpus from sourashtradictionary.com)

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install torch transformers sentencepiece sacrebleu flask pandas numpy tqdm scikit-learn
```

---

## Training

Each version has its own self-contained scripts:

```bash
# V1 — Character-level GRU baseline
python train.py

# V2 — Transformer + BPE
python train_v2.py

# V3 — T5-small fine-tuning
python train_v3.py

# V4 — T5-small + Tamil
python train_v4.py

# V5 — ByT5-small (byte-level)
python train_v5.py

# V5 Reverse — English → Sourashtra
python train_v5_reverse.py
```

---

## Web Application

```bash
python app.py
# Open http://localhost:5000
```

The app provides real-time translation between English, Tamil, and Sourashtra (rendered in native Saurashtra Unicode script), backed by the hybrid V5 model.

---

## Inference

```bash
# Hybrid translation (neural + retrieval)
python hybrid_translate.py

# Standalone inference
python inference.py
```

---

## Paper

The IEEE conference paper is in `ieee_paper/`. Key contributions:

1. Cleaned and unified Sourashtra NLP dataset (12,771 entries)
2. Progressive architecture study: character → subword → byte-level
3. Hybrid retrieval strategy combining Levenshtein, Jaccard, and prefix matching
4. Web application for real-time translation with native script rendering

---

## Citation

```bibtex
@inproceedings{sourashtra_nmt_2026,
  title     = {Progressive Neural Machine Translation for Sourashtra: From Character-Level GRU to Byte-Level Transformers},
  author    = {Pranav Aditya},
  booktitle = {IEEE Conference on},
  year      = {2026}
}
```

Original dictionary data:
```bibtex
@misc{sourashtra_dictionary,
  author = {Senthil Kumaran},
  title  = {Sourashtra Dictionary},
  year   = {2024},
  url    = {https://github.com/orsenthil/sourashtra-dictionary}
}
```

---

## License

Source code: MIT  
Dataset: Derived from the Sourashtra Dictionary project — see `sourashtra-dictionary-main/LICENSE`
