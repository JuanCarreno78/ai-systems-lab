"""RAG: recuperar fragmentos relevantes y usarlos como contexto para el LLM.

Pipeline de consulta (online), en cada pregunta:
    pregunta → embedding → búsqueda en el vector store → contexto → prompt → LLM → respuesta

CourseRAG no sabe nada de la consola: recibe una pregunta (y el historial) y devuelve
la respuesta junto con sus fuentes. Por eso otro programa puede reutilizarlo como una
capacidad más.
"""

from dataclasses import dataclass

from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from prompts import build_messages
from vector_store import SearchResult, VectorStore

NO_CONTEXT = "(No se encontraron fragmentos relevantes en los documentos del curso.)"


@dataclass
class RAGAnswer:
    text: str
    sources: list[SearchResult]  # fragmentos recuperados y enviados al LLM
    messages: list[Message]  # lo que realmente recibió el LLM
    response: LLMResponse


def format_context(results: list[SearchResult]) -> str:
    """Numera los fragmentos para que el LLM pueda citarlos: "[1] (evaluacion.md) texto"."""
    # TODO 3: convierte los resultados en el texto de contexto que leerá el LLM.
    #   - Si no hay resultados, devuelve NO_CONTEXT.
    #   - Si hay, un bloque por fragmento, numerado desde 1 y separado por una línea en blanco:
    #       [1] (evaluacion.md) El primer parcial se realizará...
    #       [2] (anuncios.md) Por la jornada institucional...
    raise NotImplementedError("Completa format_context")


class CourseRAG:
    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        results = self.store.search(query_vector, self.top_k)
        # TODO 6: descarta los resultados con score menor que self.min_score.
        #   Antes de completarlo, pregunta algo sin relación con el curso en --debug y observa
        #   qué fragmentos recibe el LLM. Elige el umbral con search.py (Paso 6 de la guía).
        return results

    def answer(self, question: str, history: list[Message] | None = None) -> RAGAnswer:
        sources = self.retrieve(question)
        messages = build_messages(history or [], question, format_context(sources))
        response = self.llm.chat(messages)
        return RAGAnswer(text=response.text, sources=sources, messages=messages, response=response)
