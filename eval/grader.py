import json
from typing import Any, Dict, List, Optional
from openai import OpenAI


def grade_quotes_relevance(
    client: OpenAI,
    question: str,
    extras: List[Dict[str, str]],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Grades extra (non-expected) citations for relevance to the question.

    Returns:
      {
        "mean": float | None,
        "per_item": [
            {"ref": str, "score": int}
        ]
      }
    """
    if not extras:
        return {"mean": None, "per_item": []}

    system = (
        "You are a strict evaluator.\n"
        "Given a QUESTION and several QUOTATIONS, score each quotation's relevance\n"
        "to answering the question.\n\n"
        "Scale: 0–5\n"
        "0 = irrelevant\n"
        "1 = very weakly related\n"
        "2 = loosely related / too general\n"
        "3 = relevant\n"
        "4 = very relevant\n"
        "5 = directly answers the question\n\n"
        "Return JSON ONLY with this exact schema:\n"
        "{\n"
        '  "per_item": [ {"ref": string, "score": number} ],\n'
        '  "mean": number\n'
        "}"
    )

    payload = {
        "question": question,
        "quotations": [
            {"ref": e["ref"], "quote": e["quote"]}
            for e in extras
        ],
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
    )

    text = resp.output_text.strip()
    data = json.loads(text)

    # Defensive normalization
    scores = []
    for it in data.get("per_item", []):
        sc = it.get("score")
        if isinstance(sc, int) and 0 <= sc <= 5:
            scores.append(sc)

    mean = (sum(scores) / len(scores)) if scores else None
    data["mean"] = mean
    return data
