from __future__ import annotations
import re
from collections.abc import Iterable

def citation_metrics(generated: Iterable[str], reference: Iterable[str]) -> dict:
    gen = {x.strip().strip("[]") for x in generated if x.strip()}
    ref = {x.strip().strip("[]") for x in reference if x.strip()}
    if not ref:
        return {"citation_precision": None if not gen else 0.0,
                "citation_recall": None, "citation_f1": None}
    if not gen:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}
    overlap = len(gen & ref)
    p, r = overlap / len(gen), overlap / len(ref)
    return {"citation_precision": p, "citation_recall": r,
            "citation_f1": 2*p*r/(p+r) if p+r else 0.0}

def citation_id_validity(generated: Iterable[str], retrieved: Iterable[str]) -> float | None:
    gen = {x.strip().strip("[]") for x in generated if x.strip()}
    ret = {x.strip() for x in retrieved if x}
    return None if not gen else len(gen & ret) / len(gen)

def abstention_heuristic(answer: str, answerable: bool) -> dict:
    markers = ("not specified", "does not specify", "no information", "cannot determine",
        "insufficient information", "documents do not", "غير موجود", "غير متاح",
        "لا توجد", "لا يوجد", "لا تحدد", "لا يذكر", "مش موجود", "مش مذكور")
    abstains = any(x in answer.casefold() for x in markers)
    return {"appears_to_abstain": abstains,
            "abstention_heuristic_match": (not abstains if answerable else abstains)}

def lexical_overlap(reference: str, generated: str) -> float | None:
    """Diagnostic only, not an answer-correctness score."""
    tokenize = lambda s: set(re.findall(r"\w+", s.casefold(), flags=re.UNICODE))
    ref, gen = tokenize(reference), tokenize(generated)
    return None if not ref else len(ref & gen) / len(ref)
