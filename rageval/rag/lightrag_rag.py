from __future__ import annotations

from collections.abc import Awaitable
from functools import partial
from pathlib import Path
from typing import Any
import json
import logging
import shutil

from rageval.config import ModelConfig
from rageval.models import RAGRunResult
from rageval.token_tracking import TokenCounter
from rageval.utils import normalize_text, slugify

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, TokenTracker
except ImportError as exc:  # pragma: no cover - dependency managed at runtime
    LightRAG = None
    QueryParam = None
    openai_complete_if_cache = None
    openai_embed = None
    EmbeddingFunc = None
    TokenTracker = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


logger = logging.getLogger(__name__)


class LightRAGSystem:
    def __init__(
        self,
        model_config: ModelConfig,
        token_counter: TokenCounter,
        workspace_root: Path,
    ) -> None:
        if IMPORT_ERROR is not None:
            raise ImportError(
                "LightRAG dependencies are unavailable. Install project dependencies first."
            ) from IMPORT_ERROR

        self.model_config = model_config
        self.token_counter = token_counter
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace_dir: Path | None = None
        self.rag: LightRAG | None = None

    async def initialize(self, doc_id: str, document: str) -> dict[str, Any]:
        self.workspace_dir = self.workspace_root / slugify(doc_id)
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        api_kwargs = {
            "api_key": self.model_config.openai_api_key,
        }
        if self.model_config.openai_base_url:
            api_kwargs["base_url"] = self.model_config.openai_base_url

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await openai_complete_if_cache(
                self.model_config.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                temperature=self.model_config.temperature,
                timeout=int(self.model_config.request_timeout_seconds),
                **api_kwargs,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.model_config.embedding_dimension,
            max_token_size=8191,
            model_name=self.model_config.embedding_model,
            func=partial(
                openai_embed.func,
                model=self.model_config.embedding_model,
                **api_kwargs,
            ),
        )

        self.rag = LightRAG(
            working_dir=str(self.workspace_dir),
            tiktoken_model_name=self.model_config.tokenizer_model,
            llm_model_func=llm_model_func,
            llm_model_name=self.model_config.llm_model,
            default_embedding_timeout=int(self.model_config.request_timeout_seconds),
            default_llm_timeout=int(self.model_config.request_timeout_seconds),
            embedding_func=embedding_func,
        )
        await self.rag.initialize_storages()

        insert_usage: dict[str, Any] = {}
        await self._run_with_optional_tracking(
            self.rag.ainsert(document),
            insert_usage,
        )
        insert_usage.setdefault(
            "estimated_document_tokens",
            self.token_counter.count_text(document),
        )
        return insert_usage

    async def query(self, question: str) -> RAGRunResult:
        if self.rag is None:
            raise RuntimeError("LightRAG graph has not been initialized for the current sample.")

        query_usage: dict[str, Any] = {}
        raw_result = await self._run_with_optional_tracking(
            self.rag.aquery(question, param=QueryParam(mode="hybrid")),
            query_usage,
        )
        answer, retrieved_context = self._extract_answer_and_context(raw_result)

        if not retrieved_context:
            context_usage: dict[str, Any] = {}
            raw_context = await self._run_with_optional_tracking(
                self.rag.aquery(
                    question,
                    param=QueryParam(mode="hybrid", only_need_context=True),
                ),
                context_usage,
            )
            retrieved_context = normalize_text(raw_context)
            if context_usage:
                query_usage["context_trace"] = context_usage

        query_usage.setdefault(
            "estimated_question_tokens",
            self.token_counter.count_text(question),
        )
        query_usage.setdefault(
            "estimated_answer_tokens",
            self.token_counter.count_text(answer),
        )
        if retrieved_context:
            query_usage.setdefault(
                "estimated_retrieved_context_tokens",
                self.token_counter.count_text(retrieved_context),
            )

        return RAGRunResult(
            answer=answer.strip(),
            retrieved_context=retrieved_context.strip(),
            token_usage=query_usage,
        )

    async def close(self) -> None:
        if self.rag is not None and hasattr(self.rag, "finalize_storages"):
            await self.rag.finalize_storages()
        if self.workspace_dir is not None and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
        self.rag = None
        self.workspace_dir = None

    async def _run_with_optional_tracking(
        self,
        awaitable: Awaitable[Any],
        usage_holder: dict[str, Any],
    ) -> Any:
        if TokenTracker is None:
            return await awaitable

        tracker = TokenTracker()
        with tracker:
            result = await awaitable
        tracker_usage = tracker.get_usage()
        if isinstance(tracker_usage, dict):
            usage_holder.update(tracker_usage)
        elif tracker_usage is not None:
            usage_holder["raw_usage"] = normalize_text(tracker_usage)
        return result

    def _extract_answer_and_context(self, raw_result: Any) -> tuple[str, str]:
        if isinstance(raw_result, str):
            return raw_result, ""

        if isinstance(raw_result, dict):
            answer = normalize_text(
                raw_result.get("response")
                or raw_result.get("answer")
                or raw_result.get("result")
                or raw_result.get("content")
            )
            context = normalize_text(
                raw_result.get("retrieved_context")
                or raw_result.get("retrieved_contexts")
                or raw_result.get("context")
                or raw_result.get("contexts")
            )
            if answer or context:
                return answer, context
            return normalize_text(raw_result), ""

        if isinstance(raw_result, (list, tuple)):
            return normalize_text(raw_result), ""

        try:
            return normalize_text(raw_result), ""
        except TypeError:
            return json.dumps(raw_result, ensure_ascii=False), ""
