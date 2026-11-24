"""Prompt-manifest invariants (``plan/m1.md`` §2.3; ``experiments/design/prompts.md`` §0, §1-§4).

Pure Python (no torch, no ``xai_dit`` import): runs on the CPU-only box.

Checked for ``experiments/design/manifest_{anima,moss_image,krea2,shared_rel}.json``:

* schema of every entry / target (``prompts.md`` §0),
* ``k >= 16`` prompts per model (``calib`` 5-10) and ~20 shared relation prompts (``rank`` + ``eval``),
* zero ``family_id`` overlap across ``calib`` / ``rank`` / ``eval`` (per file and across files),
* Anima parentheses escaped as ``\\(`` / ``\\)``,
* ``negative_prompt`` equal to the single-source standard negatives (``runtime_and_hp.md``).

Accepted file layouts: a JSON list of entries, or an object with a ``"prompts"`` list.
A manifest that is not frozen yet is *skipped*; set ``XAI_DIT_REQUIRE_MANIFESTS=1`` to make a
missing file fail (use this once M1 manifests are committed).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments" / "design"
MODELS = ("anima", "moss_image", "krea2")
MANIFESTS = MODELS + ("shared_rel",)
SPLITS = {"calib", "rank", "eval"}
REQUIRE = os.environ.get("XAI_DIT_REQUIRE_MANIFESTS") == "1"

MODEL_ALIASES = {
    "anima": {"anima"},
    "moss_image": {"moss_image", "moss-image", "moss"},
    "krea2": {"krea2", "krea_2", "krea-2"},
}
MOSS_NEG = "worst quality, bad anatomy, watermark, jpeg artifact"
STANDARD_NEGATIVES = {
    "anima": {"worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"},
    "moss_image": {MOSS_NEG, MOSS_NEG + ", 3d, realistic"},  # pure-2D items may append ", 3d, realistic"
    "krea2": {""},
}

ENTRY_FIELDS = {
    "prompt_id": str, "family_id": str, "model": (str, list), "split": str,
    "negative_prompt": (str, dict), "prompt": str, "targets": list, "seed_set": list,
}
TARGET_FIELDS = {
    "text": str, "role": str, "occurrence_idx": int, "localizable": bool,
    "neutral_sub": (str, type(None)), "vqa_question": (str, type(None)),
}
UNESCAPED_PAREN = re.compile(r"(?<!\\)[()]")
NON_TEXTUAL_ROLES = ("eos", "template", "register")  # e.g. "EOS / template tail" targets on relation probes


def _canonical_model(name: str) -> str | None:
    for canon, aliases in MODEL_ALIASES.items():
        if name.lower() in aliases:
            return canon
    return None


def load_manifest(name: str) -> list[dict]:
    path = DESIGN / f"manifest_{name}.json"
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"missing {path.relative_to(ROOT)}")
        pytest.skip(f"{path.name} not frozen yet (set XAI_DIT_REQUIRE_MANIFESTS=1 to fail instead)")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("prompts") if isinstance(data, dict) else None
    assert isinstance(entries, list) and entries, f"{path.name}: expected a non-empty list of entries"
    assert all(isinstance(e, dict) for e in entries), f"{path.name}: every entry must be an object"
    return entries


def _type_ok(value, expected) -> bool:
    if expected is int:
        return type(value) is int  # bool is not an occurrence index
    if expected is bool:
        return type(value) is bool
    return isinstance(value, expected)


def _entry_models(name: str, entry: dict) -> list[str]:
    raw = entry["model"]
    names = raw if isinstance(raw, list) else [raw]
    if name == "shared_rel" and names in (["shared"], ["all"]):
        return list(MODELS)
    return [_canonical_model(str(n)) for n in names]


# =========================================================================== schema


@pytest.mark.parametrize("name", MANIFESTS)
def test_manifest_schema(name):
    entries = load_manifest(name)
    errors = []
    for i, e in enumerate(entries):
        tag = f"{name}[{i}] {e.get('prompt_id', '?')}"
        n_before = len(errors)
        for field, typ in ENTRY_FIELDS.items():
            if field not in e:
                errors.append(f"{tag}: missing {field!r}")
            elif not _type_ok(e[field], typ):
                errors.append(f"{tag}: {field!r} has type {type(e[field]).__name__}")
        if len(errors) > n_before:
            continue
        if not e["prompt_id"].strip() or not e["family_id"].strip() or not e["prompt"].strip():
            errors.append(f"{tag}: empty prompt_id / family_id / prompt")
        if e["split"] not in SPLITS:
            errors.append(f"{tag}: split {e['split']!r} not in {sorted(SPLITS)}")
        models = _entry_models(name, e)
        if None in models or not models:
            errors.append(f"{tag}: unknown model {e['model']!r}")
        elif name != "shared_rel" and models != [name]:
            errors.append(f"{tag}: model {e['model']!r} in manifest_{name}.json")
        seeds = e["seed_set"]
        if not seeds or not all(type(s) is int for s in seeds) or len(set(seeds)) != len(seeds):
            errors.append(f"{tag}: seed_set must be a non-empty list of unique ints")
        if not e["targets"]:
            errors.append(f"{tag}: no targets")
        for j, t in enumerate(e["targets"]):
            ttag = f"{tag} target[{j}]"
            if not isinstance(t, dict):
                errors.append(f"{ttag}: not an object")
                continue
            for field, typ in TARGET_FIELDS.items():
                if field not in t:
                    errors.append(f"{ttag}: missing {field!r}")
                elif not _type_ok(t[field], typ):
                    errors.append(f"{ttag}: {field!r} has type {type(t[field]).__name__}")
            if any(f not in t or not _type_ok(t[f], TARGET_FIELDS[f]) for f in TARGET_FIELDS):
                continue
            if t["occurrence_idx"] < 0:
                errors.append(f"{ttag}: negative occurrence_idx")
            if not any(r in t["role"].lower() for r in NON_TEXTUAL_ROLES):
                count = e["prompt"].count(t["text"])
                if count <= t["occurrence_idx"]:
                    errors.append(f"{ttag}: {t['text']!r} occurs {count}x, occurrence_idx={t['occurrence_idx']}")
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("name", MANIFESTS)
def test_prompt_ids_unique(name):
    ids = [e.get("prompt_id") for e in load_manifest(name)]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"duplicate prompt_id in manifest_{name}.json: {dup}"


# =========================================================================== counts / splits


@pytest.mark.parametrize("name", MODELS)
def test_per_model_sweep_size(name):
    entries = load_manifest(name)
    by_split = defaultdict(int)
    for e in entries:
        by_split[e.get("split")] += 1
    assert len(entries) >= 16, f"{name}: k={len(entries)} < 16 (E2-1 needs disjoint 8+8 halves)"
    assert 5 <= by_split["calib"] <= 10, f"{name}: calib={by_split['calib']} not in [5, 10]"
    assert by_split["rank"] > 0 and by_split["eval"] > 0, f"{name}: rank/eval must be non-empty: {dict(by_split)}"


def test_shared_relation_set_size_and_splits():
    entries = load_manifest("shared_rel")
    assert 18 <= len(entries) <= 24, f"shared_rel: {len(entries)} prompts, expected ~20 (rank 10 + eval 10)"
    splits = {e.get("split") for e in entries}
    assert splits == {"rank", "eval"}, f"shared_rel splits {splits} (expected rank + eval only)"


@pytest.mark.parametrize("name", MANIFESTS)
def test_family_ids_disjoint_across_splits(name):
    fam_splits = defaultdict(set)
    for e in load_manifest(name):
        fam_splits[e.get("family_id")].add(e.get("split"))
    leaks = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1}
    assert not leaks, f"family_id in several splits in manifest_{name}.json: {leaks}"


def test_family_ids_disjoint_across_all_manifests():
    present = [n for n in MANIFESTS if (DESIGN / f"manifest_{n}.json").exists()]
    if not present:
        if REQUIRE:
            pytest.fail("no manifest files found")
        pytest.skip("no manifests frozen yet")
    fam_splits = defaultdict(set)
    for name in present:
        for e in load_manifest(name):
            fam_splits[e.get("family_id")].add(e.get("split"))
    leaks = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1}
    assert not leaks, f"family_id split leak across manifests: {leaks}"


# =========================================================================== text conventions


def test_anima_parentheses_escaped():
    errors = []
    for e in load_manifest("anima"):
        fields = [("prompt", e.get("prompt", ""))]
        for t in e.get("targets", []):
            fields += [("target.text", t.get("text") or ""), ("target.neutral_sub", t.get("neutral_sub") or "")]
        for field, text in fields:
            if UNESCAPED_PAREN.search(text):
                errors.append(f"{e.get('prompt_id')}: unescaped parenthesis in {field}: {text!r}")
    assert not errors, "\n".join(errors)


def test_escape_regex_sanity():
    assert UNESCAPED_PAREN.search(r"emilia \(re:zero\)") is None
    assert UNESCAPED_PAREN.search("emilia (re:zero)") is not None
    assert UNESCAPED_PAREN.search(r"emilia \(re:zero)") is not None


@pytest.mark.parametrize("name", MANIFESTS)
def test_negative_prompts_match_standard(name):
    errors = []
    for e in load_manifest(name):
        neg = e.get("negative_prompt")
        if isinstance(neg, dict):  # shared prompts: one negative per model
            for model, value in neg.items():
                canon = _canonical_model(model)
                if canon is None or value not in STANDARD_NEGATIVES[canon]:
                    errors.append(f"{e.get('prompt_id')}: negative for {model!r} = {value!r}")
        else:
            allowed = set().union(*(STANDARD_NEGATIVES[m] for m in _entry_models(name, e) if m in STANDARD_NEGATIVES))
            if neg not in allowed:
                errors.append(f"{e.get('prompt_id')}: negative_prompt {neg!r} not in {sorted(allowed)}")
    assert not errors, "\n".join(errors)
