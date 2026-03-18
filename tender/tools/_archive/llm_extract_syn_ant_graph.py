#!/usr/bin/env python3
# Archived: legacy graph-generation workflow.
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_MODEL = "gpt-4.1-mini"
ALLOWED_REL = {"synonym", "antonym"}
ALLOWED_STRENGTH = {"high", "medium", "low"}
PAIR_SPLIT_RE = re.compile(r"\s+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract synonym/antonym edges with strict verbatim-evidence constraints "
            "from per-term sentence extraction outputs."
        )
    )
    parser.add_argument("--term-sentences-dir", type=Path, required=True, help="Stage-1 output dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--batch-size", type=int, default=24, help="Sentences per LLM request")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per batch on retryable errors")
    parser.add_argument("--timeout", type=int, default=120, help="OpenAI timeout in seconds")
    parser.add_argument("--max-terms", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cached batches")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; only validate/cache structure")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def edge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["terms", "relation_type", "strength", "evidence_quote"],
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 2,
                "maxItems": 2,
            },
            "relation_type": {"type": "string", "enum": sorted(ALLOWED_REL)},
            "strength": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence_quote": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
        },
    }


def _extract_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(resp, "output", None)
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                c_text = getattr(c, "text", None)
                if isinstance(c_text, str):
                    parts.append(c_text)
                elif isinstance(c, dict) and isinstance(c.get("text"), str):
                    parts.append(c["text"])
        if parts:
            return "\n".join(parts)

    raise ValueError("Could not extract model text output")


def build_system_prompt() -> str:
    return (
        "You are a strict lexical relation extractor. "
        "Extract ONLY explicit synonym/antonym relations grounded in the provided sentences. "
        "Hard constraints: "
        "(1) Candidate terms must appear verbatim in the provided sentences. "
        "(2) evidence_quote values must be copied verbatim from provided sentences. "
        "(3) If there is no explicit equivalence/opposition evidence, output no edge. "
        "(4) Do not infer from world knowledge."
    )


def build_user_prompt(target_term: str, contexts: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"Target glossary term: {target_term}")
    lines.append("Contexts (quoted sentences, each with ID):")
    for c in contexts:
        lines.append(f"- id={c['sentence_id']} | sentence={json.dumps(c['sentence'], ensure_ascii=False)}")
    lines.append("Return JSON with key 'edges'.")
    lines.append("Each edge object must follow this schema exactly:")
    lines.append(
        '{"terms": ["<term_a>", "<term_b>"], "relation_type": "synonym|antonym", "strength": "high|medium|low", "evidence_quote": ["<verbatim sentence>"]}'
    )
    lines.append("One of terms must be exactly the target glossary term.")
    lines.append("Only include edges where evidence_quote contains explicit equivalence/opposition wording.")
    lines.append("If none exist, return {\"edges\": []}.")
    return "\n".join(lines)


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def is_retryable_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return (
        "rate" in msg
        or "429" in msg
        or "timeout" in msg
        or "temporar" in msg
        or "apierror" in name
        or "connection" in msg
    )


def normalize_pair(a: str, b: str) -> Tuple[str, str]:
    if a.casefold() <= b.casefold():
        return a, b
    return b, a


def strength_rank(value: str) -> int:
    if value == "high":
        return 3
    if value == "medium":
        return 2
    return 1


def validate_and_filter_edges(
    *,
    target_term: str,
    contexts: Sequence[Dict[str, Any]],
    raw_edges: Any,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    validated: List[Dict[str, Any]] = []

    context_sentences = [str(c.get("sentence", "")) for c in contexts]
    sentence_set = set(context_sentences)

    if not isinstance(raw_edges, list):
        return [], ["LLM payload field 'edges' is not a list"]

    for idx, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            errors.append(f"edge[{idx}] is not an object")
            continue
        required = {"terms", "relation_type", "strength", "evidence_quote"}
        missing = required - set(edge.keys())
        if missing:
            errors.append(f"edge[{idx}] missing required fields: {sorted(missing)}")
            continue
        if any(k not in required for k in edge.keys()):
            errors.append(f"edge[{idx}] has additional properties")
            continue
        if not isinstance(edge["terms"], list) or len(edge["terms"]) != 2:
            errors.append(f"edge[{idx}] terms must be an array of exactly 2 strings")
            continue
        if not all(isinstance(t, str) and t.strip() for t in edge["terms"]):
            errors.append(f"edge[{idx}] terms must be non-empty strings")
            continue
        if not isinstance(edge["evidence_quote"], list) or not edge["evidence_quote"]:
            errors.append(f"edge[{idx}] evidence_quote must be a non-empty array")
            continue

        terms = [str(t).strip() for t in edge["terms"]]
        rel = str(edge["relation_type"]).strip().lower()
        strength = str(edge["strength"]).strip().lower()
        quotes = [str(q) for q in edge["evidence_quote"] if str(q).strip()]

        if rel not in ALLOWED_REL:
            errors.append(f"edge[{idx}] invalid relation_type: {rel}")
            continue
        if strength not in ALLOWED_STRENGTH:
            errors.append(f"edge[{idx}] invalid strength: {strength}")
            continue

        target_hits = [t for t in terms if t.casefold() == target_term.casefold()]
        if not target_hits:
            errors.append(f"edge[{idx}] missing target term '{target_term}' in terms")
            continue

        if terms[0].casefold() == terms[1].casefold():
            errors.append(f"edge[{idx}] identical terms")
            continue

        other_term = terms[1] if terms[0].casefold() == target_term.casefold() else terms[0]
        if not any(other_term in sentence for sentence in context_sentences):
            errors.append(f"edge[{idx}] candidate term '{other_term}' not found verbatim in contexts")
            continue

        valid_quotes: List[str] = []
        for q in quotes:
            if q not in sentence_set:
                errors.append(f"edge[{idx}] quote not verbatim from contexts")
                continue
            if not any(target_term in q or target_term.casefold() in q.casefold() for _ in [0]):
                errors.append(f"edge[{idx}] quote missing target term")
                continue
            if other_term not in q:
                errors.append(f"edge[{idx}] quote missing candidate term '{other_term}'")
                continue
            valid_quotes.append(q)

        if not valid_quotes:
            continue

        t1, t2 = normalize_pair(terms[0], terms[1])
        validated.append(
            {
                "terms": [t1, t2],
                "relation_type": rel,
                "strength": strength,
                "evidence_quote": sorted(set(valid_quotes), key=valid_quotes.index),
            }
        )

    return validated, errors


def dedupe_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for edge in edges:
        t1, t2 = edge["terms"]
        rel = edge["relation_type"]
        strength = edge["strength"]
        quotes = edge["evidence_quote"]

        for quote in quotes:
            key = (t1.casefold(), t2.casefold(), rel, quote)
            prev = best.get(key)
            if prev is None or strength_rank(strength) > strength_rank(prev["strength"]):
                best[key] = {
                    "terms": [t1, t2],
                    "relation_type": rel,
                    "strength": strength,
                    "evidence_quote": [quote],
                }

    deduped = list(best.values())
    deduped.sort(
        key=lambda e: (
            e["terms"][0].casefold(),
            e["terms"][1].casefold(),
            e["relation_type"],
            e["evidence_quote"][0],
        )
    )
    return deduped


def load_term_files(term_sentences_dir: Path) -> List[Path]:
    index_path = term_sentences_dir / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        terms = index.get("terms") if isinstance(index, dict) else None
        if isinstance(terms, list):
            out: List[Path] = []
            for rec in terms:
                if not isinstance(rec, dict):
                    continue
                rel = rec.get("file")
                if isinstance(rel, str) and rel:
                    out.append((term_sentences_dir / rel).resolve())
            return out

    terms_dir = term_sentences_dir / "terms"
    return sorted(p.resolve() for p in terms_dir.glob("*.json") if p.is_file())


def run_llm_batch(
    *,
    client: Any,
    model: str,
    target_term: str,
    contexts: Sequence[Dict[str, Any]],
    max_retries: int,
) -> Dict[str, Any]:
    prompt = build_user_prompt(target_term=target_term, contexts=contexts)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["edges"],
        "properties": {"edges": {"type": "array", "items": edge_schema()}},
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "syn_ant_edges",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            txt = _extract_response_text(resp)
            return json.loads(txt)
        except Exception as exc:  # noqa: BLE001
            if attempt <= max_retries and is_retryable_error(exc):
                sleep_s = min(24.0, (1.8 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                time.sleep(sleep_s)
                continue
            raise


def main() -> None:
    args = parse_args()

    term_sentences_dir: Path = args.term_sentences_dir
    output_dir: Path = args.output_dir
    cache_dir = output_dir / "llm_cache"
    batches_dir = cache_dir / "batches"
    run_state_path = cache_dir / "run_state.json"
    raw_edges_path = output_dir / "raw_validated_edges.json"
    final_edges_path = output_dir / "syn_ant_graph.json"

    if not term_sentences_dir.exists() or not term_sentences_dir.is_dir():
        raise SystemExit(f"Term sentences directory not found: {term_sentences_dir}")

    term_files = load_term_files(term_sentences_dir)
    if args.max_terms > 0:
        term_files = term_files[: args.max_terms]

    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("openai package is required for LLM extraction") from exc
        client = OpenAI(timeout=int(args.timeout))

    all_validated_edges: List[Dict[str, Any]] = []
    processed_batches = 0
    skipped_batches = 0
    failed_batches = 0

    for term_idx, term_file in enumerate(term_files, start=1):
        term_payload = read_json(term_file)
        target_term = str(term_payload.get("term", "")).strip()
        if not target_term:
            continue

        sentences = term_payload.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            continue

        for batch_idx, batch in enumerate(chunked(sentences, max(1, int(args.batch_size))), start=1):
            batch_id = f"t{term_idx:04d}_b{batch_idx:04d}"
            cache_path = batches_dir / f"{batch_id}.json"

            if cache_path.exists() and not args.overwrite:
                skipped_batches += 1
                cached = read_json(cache_path)
                cached_edges = cached.get("validated_edges")
                if isinstance(cached_edges, list):
                    all_validated_edges.extend(cached_edges)
                continue

            context_payload = [
                {
                    "sentence_id": str(x.get("sentence_id", "")),
                    "source_file": str(x.get("source_file", "")),
                    "sentence": str(x.get("sentence", "")).strip(),
                }
                for x in batch
                if isinstance(x, dict) and str(x.get("sentence", "")).strip()
            ]

            batch_fingerprint = sha256_text(
                json.dumps(
                    {
                        "target_term": target_term,
                        "contexts": context_payload,
                        "model": args.model,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            if args.dry_run:
                raw_payload = {"edges": []}
                validation_errors: List[str] = []
                validated_edges: List[Dict[str, Any]] = []
                status = "dry_run"
            else:
                try:
                    raw_payload = run_llm_batch(
                        client=client,
                        model=str(args.model),
                        target_term=target_term,
                        contexts=context_payload,
                        max_retries=int(args.max_retries),
                    )
                    validated_edges, validation_errors = validate_and_filter_edges(
                        target_term=target_term,
                        contexts=context_payload,
                        raw_edges=raw_payload.get("edges"),
                    )
                    status = "ok"
                except Exception as exc:  # noqa: BLE001
                    failed_batches += 1
                    raw_payload = {"error": f"{exc.__class__.__name__}: {exc}"}
                    validation_errors = ["llm_batch_failed"]
                    validated_edges = []
                    status = "failed"

            cache_payload = {
                "batch_id": batch_id,
                "status": status,
                "target_term": target_term,
                "term_file": str(term_file),
                "batch_fingerprint": batch_fingerprint,
                "settings": {
                    "model": str(args.model),
                    "batch_size": int(args.batch_size),
                    "max_retries": int(args.max_retries),
                },
                "contexts": context_payload,
                "llm_raw": raw_payload,
                "validated_edges": validated_edges,
                "validation_errors": validation_errors,
            }
            write_json(cache_path, cache_payload)

            processed_batches += 1
            all_validated_edges.extend(validated_edges)

    write_json(raw_edges_path, all_validated_edges)

    deduped = dedupe_edges(all_validated_edges)
    write_json(final_edges_path, deduped)

    run_state = {
        "tool": "llm_extract_syn_ant_graph.py",
        "status": "ok",
        "settings": {
            "model": str(args.model),
            "batch_size": int(args.batch_size),
            "max_retries": int(args.max_retries),
            "timeout": int(args.timeout),
            "dry_run": bool(args.dry_run),
        },
        "stats": {
            "term_files": len(term_files),
            "processed_batches_now": processed_batches,
            "skipped_cached_batches": skipped_batches,
            "failed_batches": failed_batches,
            "raw_validated_edges": len(all_validated_edges),
            "deduped_edges": len(deduped),
        },
        "outputs": {
            "raw_validated_edges": str(raw_edges_path),
            "final_graph": str(final_edges_path),
        },
    }
    write_json(run_state_path, run_state)

    print(
        json.dumps(
            {
                "status": "ok",
                "processed_batches_now": processed_batches,
                "skipped_cached_batches": skipped_batches,
                "failed_batches": failed_batches,
                "raw_validated_edges": len(all_validated_edges),
                "deduped_edges": len(deduped),
                "final_graph": str(final_edges_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
