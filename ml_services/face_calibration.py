"""Threshold calibration helpers for face matching (Step 6)."""

from __future__ import annotations

from typing import Any


def suggest_threshold(
    genuine_scores: list[float],
    impostor_scores: list[float],
    *,
    target_far: float = 0.01,
) -> dict[str, Any]:
    """
    Pick a cosine-similarity threshold from labelled camera samples.

    genuine_scores: same-person match scores (higher = more similar)
    impostor_scores: different-person match scores
    target_far: maximum allowed false-accept rate on impostor scores
    """
    genuine = sorted(float(s) for s in genuine_scores if s is not None)
    impostor = sorted(float(s) for s in impostor_scores if s is not None)

    if not genuine or not impostor:
        # FAR needs impostor scores and FRR needs genuine ones: with either list empty any
        # "best" threshold is meaningless (it would just be the min / max score).
        return {
            "threshold": None,
            "min_half_total_error": None,
            "min_half_total_error_threshold": None,
            "eer": None,
            "eer_threshold": None,
            "far_at_threshold": None,
            "frr_at_threshold": None,
            "genuine_count": len(genuine),
            "impostor_count": len(impostor),
            "message": "Provide at least one genuine AND at least one impostor score.",
        }

    candidates = sorted(set(genuine + impostor))

    # Minimum half total error rate, HTER = (FAR + FRR) / 2. This is NOT the equal error rate
    # (the point where FAR == FRR); "eer" keys are kept only as backward-compatible aliases.
    best_threshold = candidates[0]
    best_hter = 1.0
    best_far = 1.0
    best_frr = 1.0

    for threshold in candidates:
        far = _rate_above(impostor, threshold)
        frr = _rate_below(genuine, threshold)
        hter = (far + frr) / 2.0
        if hter < best_hter:
            best_hter = hter
            best_threshold = threshold
            best_far = far
            best_frr = frr

    target_threshold = best_threshold
    target_far_val = best_far
    target_frr = best_frr
    for threshold in candidates:
        far = _rate_above(impostor, threshold)
        if far <= target_far:
            target_threshold = threshold
            target_far_val = far
            target_frr = _rate_below(genuine, threshold)
            break

    return {
        "threshold": round(float(target_threshold), 4),
        "min_half_total_error": round(float(best_hter), 4),
        "min_half_total_error_threshold": round(float(best_threshold), 4),
        "far_at_min_half_total_error": round(float(best_far), 4),
        "frr_at_min_half_total_error": round(float(best_frr), 4),
        # Deprecated aliases (same values as above) for existing API consumers.
        "eer_threshold": round(float(best_threshold), 4),
        "eer": round(float(best_hter), 4),
        "far_at_eer": round(float(best_far), 4),
        "frr_at_eer": round(float(best_frr), 4),
        "far_at_threshold": round(float(target_far_val), 4),
        "frr_at_threshold": round(float(target_frr), 4),
        "target_far": float(target_far),
        "genuine_count": len(genuine),
        "impostor_count": len(impostor),
        "message": (
            f"Suggested threshold {target_threshold:.4f} "
            f"(FAR={target_far_val:.2%}, FRR={target_frr:.2%} at target FAR {target_far:.2%})."
        ),
    }


def _rate_above(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def _rate_below(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s < threshold) / len(scores)
