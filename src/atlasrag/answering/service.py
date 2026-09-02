from dataclasses import dataclass
from typing import Protocol

from atlasrag.answering.context import Citation, ContextBuilder, RetrievedContext
from atlasrag.retrieval.vector_index import SearchHit


class Retriever(Protocol):
    """Boundary implemented by semantic retrieval backends."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float = -1.0,
    ) -> tuple[SearchHit, ...]: ...


class AnswerGenerator(Protocol):
    """Boundary implemented by local or hosted language models."""

    def generate(
        self,
        *,
        query: str,
        context: RetrievedContext,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Generated answer and the exact sources made available to it."""

    query: str
    answer: str
    context: RetrievedContext

    @property
    def citations(self) -> tuple[Citation, ...]:
        return self.context.citations


class AnswerGenerationError(RuntimeError):
    """Raised when an answer generator returns unusable output."""


class RagService:
    """Coordinate retrieval, context construction, and answer generation."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: AnswerGenerator,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._context_builder = context_builder or ContextBuilder()

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float = -1.0,
    ) -> AnswerResult:
        """Answer one query using only the context selected by retrieval."""
        if not query.strip():
            raise ValueError("query must not be blank")

        hits = self._retriever.search(
            query,
            top_k=top_k,
            minimum_score=minimum_score,
        )
        context = self._context_builder.build(hits)
        answer = self._generator.generate(query=query, context=context)
        if not answer.strip():
            raise AnswerGenerationError("answer generator returned blank output")

        return AnswerResult(query=query, answer=answer, context=context)
