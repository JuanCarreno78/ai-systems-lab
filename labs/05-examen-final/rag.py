"""RAG: recuperar fragmentos relevantes y usarlos como contexto para el LLM.

Pipeline de consulta (online), en cada pregunta:
    pregunta → embedding → búsqueda en el vector store → contexto → prompt → LLM → respuesta

CourseRAG no sabe nada de la consola: recibe una pregunta (y el historial) y devuelve
la respuesta junto con sus fuentes. Por eso otro programa puede reutilizarlo como una
capacidad más.

Examen final: la búsqueda recibe los niveles que permite el rol y filtra antes de armar el
contexto. También avisa si la mejor respuesta estaba en un nivel que el rol no puede ver, para
pedir login sin mostrar el contenido.
"""

from dataclasses import dataclass

from embeddings import Embedder
from llm_client import LLMClient, LLMResponse, Message
from prompts import build_rag_messages
from vector_store import SearchResult, VectorStore

NO_CONTEXT = "(No se encontraron fragmentos relevantes en los documentos del curso.)"


@dataclass
class Retrieval:
    results: list[SearchResult]  # fragmentos permitidos que superan el umbral
    bloqueado: str | None  # nivel de un fragmento no permitido que habría entrado al top-k sin el filtro


@dataclass
class RAGAnswer:
    text: str
    sources: list[SearchResult]  # fragmentos recuperados y enviados al LLM
    messages: list[Message]  # lo que realmente recibió el LLM
    response: LLMResponse | None  # None si no se llamó al LLM
    bloqueado: str | None = None


def format_context(results: list[SearchResult]) -> str:
    """Numera los fragmentos para que el LLM pueda citarlos: "[1] (evaluacion.md) texto"."""
    if not results:
        return NO_CONTEXT
    return "\n\n".join(
        f"[{i}] ({result.chunk.source}) {result.chunk.text}" for i, result in enumerate(results, 1)
    )


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

    def retrieve(self, question: str, niveles: set[str]) -> Retrieval:
        query_vector = self.embedder.embed_query(question)
        results = [r for r in self.store.search(query_vector, self.top_k, niveles) if r.score >= self.min_score]
        # ¿Sin el filtro habría llegado algún fragmento no permitido? Solo se usa su nivel, nunca su texto.
        sin_filtro = self.store.search(query_vector, self.top_k)
        bloqueado = next(
            (r.chunk.nivel for r in sin_filtro if r.chunk.nivel not in niveles and r.score >= self.min_score), None
        )
        return Retrieval(results, bloqueado)

    def answer(
        self, question: str, history: list[Message], niveles: set[str], usuario: str | None = None
    ) -> RAGAnswer:
        retrieval = self.retrieve(question, niveles)
        if not retrieval.results and retrieval.bloqueado:
            # Solo hay respuesta en documentos que el rol no puede ver: no se llama al LLM.
            return RAGAnswer("", [], [], None, retrieval.bloqueado)
        messages = build_rag_messages(history, question, format_context(retrieval.results), usuario)
        response = self.llm.chat(messages)
        return RAGAnswer(response.text, retrieval.results, messages, response, retrieval.bloqueado)
