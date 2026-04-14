from __future__ import annotations

from typing import Any
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rageval.config import ModelConfig, VectorRAGConfig
from rageval.models import RAGRunResult
from rageval.token_tracking import TokenCounter, extract_langchain_usage


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise QA assistant for a retrieval experiment. "
    "Answer only from the provided context. "
    "If the document does not contain enough evidence, say that the answer "
    "cannot be determined from the provided context."
)

USER_PROMPT_TEMPLATE = """Question:
{question}

Retrieved Context:
{context}

Answer:"""


class VectorRAGSystem:
    def __init__(
        self,
        model_config: ModelConfig,
        rag_config: VectorRAGConfig,
        token_counter: TokenCounter,
    ) -> None:
        self.model_config = model_config
        self.rag_config = rag_config
        self.token_counter = token_counter
        self.vectorstore: FAISS | None = None
        self.chunk_token_counts: list[int] = []

        common_kwargs = {
            "api_key": self.model_config.openai_api_key,
            "base_url": self.model_config.openai_base_url,
            "max_retries": self.model_config.max_retries,
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap,
        )
        self.embedding_model = OpenAIEmbeddings(
            model=self.model_config.embedding_model,
            timeout=self.model_config.request_timeout_seconds,
            **common_kwargs,
        )
        self.chat_model = ChatOpenAI(
            model=self.model_config.llm_model,
            temperature=self.model_config.temperature,
            timeout=self.model_config.request_timeout_seconds,
            **common_kwargs,
        )

    def reset(self) -> None:
        self.vectorstore = None
        self.chunk_token_counts = []

    def index_document(self, document: str, doc_id: str) -> dict[str, Any]:
        self.reset()
        documents = self.text_splitter.create_documents(
            texts=[document],
            metadatas=[{"doc_id": doc_id}],
        )
        self.chunk_token_counts = [
            self.token_counter.count_text(chunk.page_content) for chunk in documents
        ]
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        logger.info(
            "VectorRAG indexed doc_id=%s with %s chunks",
            doc_id,
            len(documents),
        )
        return {
            "embedded_chunk_count": len(self.chunk_token_counts),
            "estimated_embedding_input_tokens": sum(self.chunk_token_counts),
        }

    def answer(self, question: str) -> RAGRunResult:
        if self.vectorstore is None:
            raise RuntimeError("VectorRAG index has not been built for the current sample.")

        retrieved_docs = self.vectorstore.similarity_search(question, k=self.rag_config.top_k)
        retrieved_context = "\n\n".join(
            f"[Chunk {index}]\n{doc.page_content}"
            for index, doc in enumerate(retrieved_docs, start=1)
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            context=retrieved_context,
        )
        response = self.chat_model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        answer = response.content if isinstance(response.content, str) else str(response.content)
        token_usage = extract_langchain_usage(response)
        token_usage.update(
            {
                "embedded_chunk_count": len(self.chunk_token_counts),
                "estimated_embedding_input_tokens": sum(self.chunk_token_counts),
                "estimated_question_tokens": self.token_counter.count_text(question),
                "estimated_retrieved_context_tokens": self.token_counter.count_text(
                    retrieved_context
                ),
                "estimated_answer_tokens": self.token_counter.count_text(answer),
            }
        )

        return RAGRunResult(
            answer=answer.strip(),
            retrieved_context=retrieved_context.strip(),
            token_usage=token_usage,
        )
