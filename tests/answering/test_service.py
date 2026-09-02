import pytest

from atlasrag.answering.context import ContextBuilder, ContextConfig, RetrievedContext
from atlasrag.answering.service import AnswerGenerationError, RagService
from atlasrag.retrieval.vector_index import SearchHit

from .test_context import _search_hit


class FakeRetriever:
    def __init__(self, hits: tuple[SearchHit, ...]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, float]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float = -1.0,
    ) -> tuple[SearchHit, ...]:
        self.calls.append((query, top_k, minimum_score))
        return self.hits


class FakeGenerator:
    def __init__(self, answer: str = "Report the incident immediately [1].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, RetrievedContext]] = []

    def generate(self, *, query: str, context: RetrievedContext) -> str:
        self.calls.append((query, context))
        return self.answer


def test_rag_service_coordinates_retrieval_context_and_generation() -> None:
    retriever = FakeRetriever((_search_hit(0, "Report incidents immediately."),))
    generator = FakeGenerator()
    service = RagService(
        retriever=retriever,
        generator=generator,
        context_builder=ContextBuilder(ContextConfig(max_chunks=1)),
    )

    result = service.ask("What do I do?", top_k=3, minimum_score=0.4)

    assert retriever.calls == [("What do I do?", 3, 0.4)]
    assert generator.calls == [("What do I do?", result.context)]
    assert result.answer == "Report the incident immediately [1]."
    assert result.query == "What do I do?"
    assert result.citations == result.context.citations
    assert result.citations[0].label == "[1]"


@pytest.mark.parametrize("query", ["", "   "])
def test_rag_service_rejects_blank_query_before_retrieval(query: str) -> None:
    retriever = FakeRetriever(())
    generator = FakeGenerator()

    with pytest.raises(ValueError, match="query must not be blank"):
        RagService(retriever=retriever, generator=generator).ask(query)

    assert retriever.calls == []
    assert generator.calls == []


def test_rag_service_rejects_blank_generated_answer() -> None:
    retriever = FakeRetriever(())
    generator = FakeGenerator("   ")

    with pytest.raises(
        AnswerGenerationError,
        match="answer generator returned blank output",
    ):
        RagService(retriever=retriever, generator=generator).ask("Question")
