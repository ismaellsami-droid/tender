#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from openai import OpenAI


class BaseCorpusEngine:
    def __init__(
        self,
        *,
        corpus_id: str,
        model: str,
        vector_store_id: str,
    ) -> None:
        self.corpus_id = corpus_id
        self.model = model
        self.vector_store_id = vector_store_id
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client
