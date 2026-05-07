"""
Evaluation metrics for Sourashtra Translation
===============================================
BLEU, chrF, Character Error Rate, Word Error Rate, and Exact Match.
All metrics computed without external dependencies for portability.
"""
import re
import math
from collections import Counter


def _tokenize(text):
    """Simple tokenizer: lowercase, split on whitespace and punctuation."""
    return text.lower().split()


def _char_tokenize(text):
    """Character-level tokenizer."""
    return list(text.lower().strip())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLEU Score (sentence-level and corpus-level)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_ngrams(tokens, n):
    """Compute n-gram counts."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4, smooth=True):
    """
    Compute sentence-level BLEU score.

    Args:
        reference: reference string
        hypothesis: hypothesis/prediction string
        max_n: maximum n-gram order (default 4)
        smooth: use smoothing (add-1) for sentence-level
    Returns:
        BLEU score (0-100)
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if len(hyp_tokens) == 0:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)

        clipped = sum(min(count, ref_ngrams.get(ngram, 0))
                      for ngram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        elif smooth:
            precisions.append((clipped + 1) / (total + 1))
        else:
            precisions.append(clipped / total if clipped > 0 else 0.0)

    # Check for zero precisions
    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * math.exp(log_avg) * 100


def corpus_bleu(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU score.

    Args:
        references: list of reference strings
        hypotheses: list of hypothesis strings
        max_n: maximum n-gram order
    Returns:
        BLEU score (0-100)
    """
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = _tokenize(ref)
        hyp_tokens = _tokenize(hyp)

        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = compute_ngrams(ref_tokens, n)
            hyp_ngrams = compute_ngrams(hyp_tokens, n)

            for ngram, count in hyp_ngrams.items():
                clipped_counts[n-1] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[n-1] += sum(hyp_ngrams.values())

    # Precisions
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / max(hyp_len, 1))

    return bp * math.exp(log_avg) * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# chrF Score (character F-score)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chrf_score(reference, hypothesis, max_n=6, beta=2.0):
    """
    Compute chrF score (character n-gram F-score).

    Args:
        reference: reference string
        hypothesis: hypothesis string
        max_n: max character n-gram order
        beta: F-score beta (default 2.0 = recall-weighted)
    Returns:
        chrF score (0-100)
    """
    ref_chars = _char_tokenize(reference)
    hyp_chars = _char_tokenize(hypothesis)

    if len(hyp_chars) == 0 and len(ref_chars) == 0:
        return 100.0
    if len(hyp_chars) == 0 or len(ref_chars) == 0:
        return 0.0

    precisions = []
    recalls = []

    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_chars, n)
        hyp_ngrams = compute_ngrams(hyp_chars, n)

        if not hyp_ngrams or not ref_ngrams:
            continue

        clipped = sum(min(count, ref_ngrams.get(ngram, 0))
                      for ngram, count in hyp_ngrams.items())

        precision = clipped / sum(hyp_ngrams.values()) if sum(hyp_ngrams.values()) > 0 else 0
        recall = clipped / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    if avg_precision + avg_recall == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * avg_precision * avg_recall / (beta_sq * avg_precision + avg_recall)

    return f_score * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character Error Rate (CER) & Word Error Rate (WER)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _levenshtein(s1, s2):
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(reference, hypothesis):
    """Compute Character Error Rate (CER). Returns 0-100."""
    ref_chars = list(reference.lower().strip())
    hyp_chars = list(hypothesis.lower().strip())

    if len(ref_chars) == 0:
        return 100.0 if len(hyp_chars) > 0 else 0.0

    distance = _levenshtein(ref_chars, hyp_chars)
    return (distance / len(ref_chars)) * 100


def word_error_rate(reference, hypothesis):
    """Compute Word Error Rate (WER). Returns 0-100."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0

    distance = _levenshtein(ref_words, hyp_words)
    return (distance / len(ref_words)) * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exact Match Accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exact_match(reference, hypothesis):
    """Check if prediction exactly matches reference (case-insensitive)."""
    return reference.lower().strip() == hypothesis.lower().strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batch Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_all(references, hypotheses):
    """
    Compute all metrics on a list of reference/hypothesis pairs.

    Returns:
        dict with all metric scores
    """
    n = len(references)
    assert n == len(hypotheses), "Mismatched lengths"

    # Corpus-level BLEU
    bleu = corpus_bleu(references, hypotheses)

    # Averages of sentence-level metrics
    bleu_scores = [sentence_bleu(r, h) for r, h in zip(references, hypotheses)]
    chrf_scores = [chrf_score(r, h) for r, h in zip(references, hypotheses)]
    cer_scores = [character_error_rate(r, h) for r, h in zip(references, hypotheses)]
    wer_scores = [word_error_rate(r, h) for r, h in zip(references, hypotheses)]
    exact_matches = [exact_match(r, h) for r, h in zip(references, hypotheses)]

    results = {
        "corpus_bleu": round(bleu, 2),
        "avg_sentence_bleu": round(sum(bleu_scores) / n, 2),
        "avg_chrf": round(sum(chrf_scores) / n, 2),
        "avg_cer": round(sum(cer_scores) / n, 2),
        "avg_wer": round(sum(wer_scores) / n, 2),
        "exact_match_accuracy": round(sum(exact_matches) / n * 100, 2),
        "num_samples": n,
    }

    return results


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Corpus BLEU:           {metrics['corpus_bleu']:.2f}")
    print(f"  Avg Sentence BLEU:     {metrics['avg_sentence_bleu']:.2f}")
    print(f"  Avg chrF:              {metrics['avg_chrf']:.2f}")
    print(f"  Avg CER:               {metrics['avg_cer']:.2f}%")
    print(f"  Avg WER:               {metrics['avg_wer']:.2f}%")
    print(f"  Exact Match Accuracy:  {metrics['exact_match_accuracy']:.2f}%")
    print(f"  Samples:               {metrics['num_samples']}")
    print(f"{'='*50}")
