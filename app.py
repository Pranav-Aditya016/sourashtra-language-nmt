"""
Sourashtra Translation Web App
================================
Flask backend serving:
  1. Dictionary lookup: English/Tamil → Sourashtra (fuzzy search)
  2. Neural translation: Sourashtra → English (V5 ByT5 model)

Usage:
    python app.py
    → Open http://localhost:5000
"""
import os
import time
import pandas as pd
import torch
from flask import Flask, request, jsonify, send_from_directory
from config_v5_reverse import ConfigV5Reverse

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "cleaned_data")
UNIFIED_FILE = os.path.join(DATA_DIR, "unified_full_dataset.csv")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "website")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_v5", "best_model_v5")
REVERSE_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints_v5_reverse", "best_model_v5_reverse"
)

# V5 model generation settings
NUM_BEAMS = 4
MAX_GENERATE_LEN = 64
MAX_SOURCE_LEN = 128

# V5-REVERSE model generation settings (English/Tamil → Sourashtra)
REV_NUM_BEAMS = ConfigV5Reverse.NUM_BEAMS
REV_MAX_GENERATE_LEN = ConfigV5Reverse.MAX_GENERATE_LEN
REV_MAX_SOURCE_LEN = ConfigV5Reverse.MAX_SOURCE_LEN

# ═══════════════════════════════════════════════════════════
# SIMILARITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def char_ngrams(word: str, n: int = 3) -> set:
    padded = f"^{word.lower().strip()}$"
    return set(padded[i:i+n] for i in range(len(padded) - n + 1))


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=None)

entries = []           # Full dictionary
english_lower = []     # Pre-computed lowercase English
tamil_lower = []       # Pre-computed lowercase Tamil
roman_lower = []       # Pre-computed lowercase Roman
en_trigrams = []       # Pre-computed English trigrams
en_bigrams = []        # Pre-computed English bigrams

# Neural model state (Sourashtra → English, loaded at startup)
neural_model = None
neural_tokenizer = None
neural_device = None
neural_available = False

# Reverse neural model state (English/Tamil → Sourashtra, optional)
reverse_neural_model = None
reverse_neural_tokenizer = None
reverse_neural_device = None
reverse_neural_available = False


def load_resources():
    """Load dictionary from unified dataset at startup."""
    global entries, english_lower, tamil_lower, roman_lower
    global en_trigrams, en_bigrams

    print("[LOAD] Building dictionary from unified dataset...")
    t0 = time.time()

    df = pd.read_csv(UNIFIED_FILE, encoding="utf-8")

    seen = set()
    for _, row in df.iterrows():
        sr_script = str(row.get("sourashtra_word", "")).strip()
        roman = str(row.get("roman_readable", "")).strip()
        english = str(row.get("meaning_english", "")).strip()
        tamil = str(row.get("meaning_tamil", "")).strip()
        category = str(row.get("category", "")).strip()

        if not sr_script or sr_script == "nan" or not english or english == "nan":
            continue

        key = (english.lower(), sr_script)
        if key in seen:
            continue
        seen.add(key)

        entries.append({
            "sourashtra_script": sr_script,
            "roman": roman if roman != "nan" else "",
            "english": english,
            "tamil": tamil if tamil != "nan" else "",
            "category": category if category != "nan" else "",
        })

    english_lower = [e["english"].lower() for e in entries]
    tamil_lower = [e["tamil"].lower() for e in entries]
    roman_lower = [e["roman"].lower() for e in entries]
    en_trigrams = [char_ngrams(e, 3) for e in english_lower]
    en_bigrams = [char_ngrams(e, 2) for e in english_lower]

    elapsed = time.time() - t0
    cats = set(e["category"] for e in entries if e["category"])
    print(f"  Dictionary: {len(entries):,} entries, {len(cats)} categories ({elapsed:.1f}s)")
    print("[LOAD] Dictionary ready!")


def load_neural_model():
    """Load V5 ByT5 neural model for Sourashtra → English translation."""
    global neural_model, neural_tokenizer, neural_device, neural_available

    if not os.path.exists(MODEL_DIR):
        print(f"[NEURAL] Model directory not found: {MODEL_DIR}")
        print("[NEURAL] Running in dictionary-only mode.")
        return

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print("[NEURAL] Loading V5 ByT5-small model...")
        t0 = time.time()

        neural_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        neural_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(neural_device)
        neural_model.eval()
        neural_available = True

        elapsed = time.time() - t0
        print(f"[NEURAL] Model loaded on {neural_device} in {elapsed:.1f}s")
        print(f"[NEURAL] Sourashtra → English translation is ACTIVE")
    except Exception as e:
        print(f"[NEURAL] Failed to load model: {e}")
        print("[NEURAL] Running in dictionary-only mode.")
        neural_available = False


def load_reverse_neural_model():
    """Load V5-Reverse ByT5 neural model for English/Tamil → Sourashtra translation."""
    global reverse_neural_model, reverse_neural_tokenizer, reverse_neural_device, reverse_neural_available

    if not os.path.exists(REVERSE_MODEL_DIR):
        print(f"[NEURAL-REV] Model directory not found: {REVERSE_MODEL_DIR}")
        print("[NEURAL-REV] Reverse model not available. Falling back to dictionary for EN/TA → Sourashtra.")
        return

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print("[NEURAL-REV] Loading V5-Reverse ByT5-small model...")
        t0 = time.time()

        reverse_neural_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reverse_neural_tokenizer = AutoTokenizer.from_pretrained(REVERSE_MODEL_DIR)
        reverse_neural_model = AutoModelForSeq2SeqLM.from_pretrained(REVERSE_MODEL_DIR).to(reverse_neural_device)
        reverse_neural_model.eval()
        reverse_neural_available = True

        elapsed = time.time() - t0
        print(f"[NEURAL-REV] Reverse model loaded on {reverse_neural_device} in {elapsed:.1f}s")
        print("[NEURAL-REV] English/Tamil → Sourashtra translation is ACTIVE")
    except Exception as e:
        print(f"[NEURAL-REV] Failed to load reverse model: {e}")
        print("[NEURAL-REV] Falling back to dictionary for EN/TA → Sourashtra.")
        reverse_neural_available = False


def neural_translate(text, category=None):
    """
    Translate Sourashtra (Roman script) → English using V5 ByT5 model.
    Returns the English translation string.
    """
    # Format input the same way as training (from data_loader_v5.format_sr_to_en)
    if category and str(category) != "nan" and category.strip():
        input_text = f"translate Sourashtra [{category.strip()}] to English: {text.strip()}"
    else:
        input_text = f"translate Sourashtra to English: {text.strip()}"

    inputs = neural_tokenizer(
        input_text, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SOURCE_LEN
    ).to(neural_device)

    with torch.no_grad():
        outputs = neural_model.generate(
            **inputs,
            max_length=MAX_GENERATE_LEN,
            num_beams=NUM_BEAMS,
            length_penalty=1.0,
        )

    decoded = neural_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


def reverse_neural_translate(text, lang="english", category=None):
    """
    Translate English/Tamil → Sourashtra (Roman script) using V5-Reverse ByT5 model.
    Returns the Sourashtra romanized translation string.
    """
    lang = (lang or "english").lower()

    # Format input the same way as training (see data_loader_v5_reverse.format_*)
    if lang.startswith("ta"):
        if category and str(category) != "nan" and category.strip():
            input_text = f"translate Tamil [{category.strip()}] to Sourashtra: {text.strip()}"
        else:
            input_text = f"translate Tamil to Sourashtra: {text.strip()}"
    else:
        if category and str(category) != "nan" and category.strip():
            input_text = f"translate English [{category.strip()}] to Sourashtra: {text.strip()}"
        else:
            input_text = f"translate English to Sourashtra: {text.strip()}"

    inputs = reverse_neural_tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=REV_MAX_SOURCE_LEN,
    ).to(reverse_neural_device)

    with torch.no_grad():
        outputs = reverse_neural_model.generate(
            **inputs,
            max_length=REV_MAX_GENERATE_LEN,
            num_beams=REV_NUM_BEAMS,
            length_penalty=ConfigV5Reverse.LENGTH_PENALTY,
        )

    decoded = reverse_neural_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


def search_entries(query: str, lang: str = "auto", top_k: int = 10):
    """
    Search dictionary for English or Tamil → Sourashtra matches.
    Returns ranked results with confidence scores.
    """
    q = query.lower().strip()
    if not q:
        return []

    is_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in q)
    if lang == "auto":
        lang = "tamil" if is_tamil else "english"

    q_tri = char_ngrams(q, 3)
    q_bi = char_ngrams(q, 2)
    results = []

    for i in range(len(entries)):
        target = tamil_lower[i] if lang == "tamil" else english_lower[i]
        if not target:
            continue

        # Exact match
        if target == q:
            results.append((i, 1.0))
            continue

        # Substring containment
        if q in target or target in q:
            ratio = min(len(q), len(target)) / max(len(q), len(target))
            results.append((i, 0.5 + 0.4 * ratio))
            continue

        if lang == "english":
            # Enhanced similarity for English
            max_len = max(len(q), len(target))
            if max_len == 0:
                continue

            j3 = jaccard_similarity(q_tri, en_trigrams[i])
            j2 = jaccard_similarity(q_bi, en_bigrams[i])
            jac = 0.55 * j3 + 0.45 * j2

            lev = 1.0 - levenshtein_distance(q, target) / max_len

            pfx = 0
            for a, b in zip(q, target):
                if a == b:
                    pfx += 1
                else:
                    break
            pfx_ratio = pfx / max_len

            score = 0.45 * jac + 0.45 * lev + 0.10 * pfx_ratio
            if score > 0.25:
                results.append((i, score))

    results.sort(key=lambda x: -x[1])

    # Deduplicate by script form
    seen_scripts = set()
    unique = []
    for idx, conf in results:
        script = entries[idx]["sourashtra_script"]
        if script not in seen_scripts:
            seen_scripts.add(script)
            unique.append({
                **entries[idx],
                "confidence": round(conf, 4),
            })
        if len(unique) >= top_k:
            break

    return unique


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/translate", methods=["POST"])
def translate():
    """
    Translate English/Tamil → Sourashtra.
    
    Request: {"text": "milk", "lang": "auto"|"english"|"tamil"}
    Response: {"results": [...], "time_ms": 5, "detected_lang": "english"}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    lang = data.get("lang", "auto")
    t0 = time.time()

    is_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in text)
    detected = "tamil" if is_tamil else "english"

    results = search_entries(text, lang=lang, top_k=10)
    elapsed_ms = round((time.time() - t0) * 1000)

    return jsonify({
        "results": results,
        "time_ms": elapsed_ms,
        "input": text,
        "detected_lang": detected,
        "total_matches": len(results),
    })


@app.route("/api/stats")
def stats():
    cats = sorted(set(e["category"] for e in entries if e["category"]))
    return jsonify({
        "dictionary_size": len(entries),
        "categories": cats,
        "category_count": len(cats),
        "project": "Sourashtra-English Neural Machine Translation",
        "versions": {
            "V1": {"name": "Char-GRU Seq2Seq", "em": 0.42, "params": "~1M"},
            "V2": {"name": "Transformer + BPE", "em": 2.56, "params": "~5M"},
            "V3": {"name": "T5-small (EN only)", "em": 6.01, "params": "60M"},
            "V3H": {"name": "T5 + Retrieval", "em": 7.47, "params": "60M"},
            "V4": {"name": "T5 + Tamil", "em": 5.80, "params": "60M"},
            "V4H": {"name": "T5 + Tamil + Retrieval", "em": 7.68, "params": "60M"},
            "V5": {"name": "ByT5-small (Byte-level)", "em": 9.25, "params": "300M"},
            "V5H": {"name": "ByT5 + Enhanced Retrieval", "em": 9.61, "params": "300M"},
        },
    })


@app.route("/api/dictionary", methods=["GET"])
def browse_dictionary():
    query = request.args.get("q", "").lower().strip()
    category = request.args.get("category", "").strip()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    filtered = entries
    if query:
        filtered = [e for e in filtered
                     if query in e["english"].lower()
                     or query in e["tamil"].lower()
                     or query in e["roman"].lower()
                     or query in e["sourashtra_script"]]
    if category:
        filtered = [e for e in filtered if e["category"].lower() == category.lower()]

    total = len(filtered)
    start = (page - 1) * per_page
    items = filtered[start:start + per_page]
    categories = sorted(set(e["category"] for e in entries if e["category"]))

    return jsonify({
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "categories": categories,
    })


@app.route("/api/neural-translate", methods=["POST"])
def neural_translate_api():
    """
    Neural translation: Sourashtra (Roman) -> English.

    Request:  {"text": "paal", "category": "food"} (category is optional)
    Response: {"translation": "milk", "input": "paal", "model": "V5 ByT5-small", "time_ms": 120}
    """
    if not neural_available:
        return jsonify({
            "error": "Neural model not available. Server is running in dictionary-only mode.",
            "neural_available": False,
        }), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    category = data.get("category", "")
    t0 = time.time()

    try:
        translation = neural_translate(text, category)
        elapsed_ms = round((time.time() - t0) * 1000)

        # Also try to find the word in dictionary for extra context
        dict_matches = search_entries(text, lang="english", top_k=3)

        return jsonify({
            "translation": translation,
            "input": text,
            "category_hint": category,
            "model": "V5 ByT5-small (300M params)",
            "time_ms": elapsed_ms,
            "neural_available": True,
            "dict_matches": dict_matches,
            "mode": "neural",
            "source_lang": "sourashtra",
            "target_lang": "english",
            "direction": "sr_to_en",
        })
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


@app.route("/api/reverse-neural-translate", methods=["POST"])
def reverse_neural_translate_api():
    """
    Neural translation (with dictionary fallback): English/Tamil -> Sourashtra (Roman).

    Request:  {"text": "milk", "lang": "auto"|"english"|"tamil", "category": "food"}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    lang = data.get("lang", "auto")
    category = data.get("category", "")

    # Language detection (for 'auto')
    is_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in text)
    detected = "tamil" if is_tamil else "english"
    if lang == "auto":
        lang = detected
    else:
        lang = str(lang).lower()

    t0 = time.time()

    # If reverse model not available, fall back to dictionary search
    if not reverse_neural_available:
        results = search_entries(text, lang=lang, top_k=10)
        elapsed_ms = round((time.time() - t0) * 1000)

        return jsonify({
            "results": results,
            "time_ms": elapsed_ms,
            "input": text,
            "detected_lang": detected,
            "total_matches": len(results),
            "mode": "dictionary",
            "neural_available": False,
            "source_lang": lang,
            "target_lang": "sourashtra",
            "direction": f"{lang}_to_sr",
        })

    try:
        translation = reverse_neural_translate(text, lang=lang, category=category)
        elapsed_ms = round((time.time() - t0) * 1000)

        # Also try to find the word in dictionary (for cross-reference) based on source language
        dict_matches = search_entries(text, lang=lang, top_k=3)
        direction = "en_to_sr" if lang == "english" else "ta_to_sr"

        return jsonify({
            "translation": translation,
            "input": text,
            "category_hint": category,
            "model": "V5-Reverse ByT5-small (300M params)",
            "time_ms": elapsed_ms,
            "neural_available": True,
            "mode": "neural",
            "dict_matches": dict_matches,
            "source_lang": lang,
            "target_lang": "sourashtra",
            "direction": direction,
        })
    except Exception as e:
        return jsonify({"error": f"Reverse translation failed: {str(e)}"}), 500


@app.route("/api/model-status")
def model_status():
    """Check if neural model is loaded and available."""
    return jsonify({
        "neural_available": neural_available,
        "model": "V5 ByT5-small" if neural_available else None,
        "device": str(neural_device) if neural_available else None,
        "reverse_neural_available": reverse_neural_available,
        "reverse_model": "V5-Reverse ByT5-small" if reverse_neural_available else None,
        "reverse_device": str(reverse_neural_device) if reverse_neural_available else None,
    })


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_resources()
    load_neural_model()
    load_reverse_neural_model()
    print("\n[DEBUG] Registered Flask routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule:40s} → {rule.endpoint:25s} methods={rule.methods}")
    print("\n" + "=" * 50)
    print("  Sourashtra Translator — http://localhost:5000")
    print("  Dictionary: English/Tamil → Sourashtra")
    if neural_available:
        print("  Neural AI (SR→EN): AVAILABLE (V5 ByT5)")
    else:
        print("  Neural AI (SR→EN): NOT AVAILABLE (dictionary-only mode)")
    if reverse_neural_available:
        print("  Neural AI (EN/TA→SR): AVAILABLE (V5-Reverse ByT5)")
    else:
        print("  Neural AI (EN/TA→SR): NOT AVAILABLE (falls back to dictionary)")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
