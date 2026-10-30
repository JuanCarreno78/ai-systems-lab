"""Capacidades del asistente expresadas como tools.

El RAG del Lab 02 se convierte aquí en una tool más (buscar_documentos). Solo hace la
RECUPERACIÓN: devuelve los fragmentos numerados con su fuente, y el LLM que la pidió
redacta la respuesta. Para el agente no hay diferencia entre buscar en documentos,
consultar un servidor MCP o ejecutar una función local: todas son `Tool`.

Uso (sin LLM ni API key; requiere el índice de ingest.py):
    uv run python labs/05-examen-final/capabilities.py "¿Cuándo es el primer parcial?"
"""

import asyncio
import sys
from typing import Any

from rag import CourseRAG, format_context
from tools import Tool


AVISO_RESTRINGIDO = (
    "(La respuesta está en documentos restringidos que esta sesión no puede ver. "
    "Dile al usuario que necesita iniciar sesión con /login o que su rol no tiene acceso; no inventes el dato.)"
)


def rag_tool(rag: CourseRAG, niveles: set[str]) -> Tool:
    # TODO 1: la recuperación del Lab 02 como tool. Solo recupera: la respuesta la redacta el agente.
    # Examen: solo busca en los niveles que permite el rol de la sesión.
    async def run(arguments: dict[str, Any]) -> str:
        retrieval = rag.retrieve(arguments["consulta"], niveles)
        if not retrieval.results and retrieval.bloqueado:
            return AVISO_RESTRINGIDO
        return format_context(retrieval.results)

    return Tool(
        name="buscar_documentos",
        description="Busca en los documentos oficiales del curso: programa, docente, horarios, "
        "evaluación y distribución de la nota, fechas de parciales, políticas (entregas tarde, uso de "
        "IA, trabajo en parejas), anuncios y preguntas frecuentes de los laboratorios. Devuelve "
        "fragmentos numerados con su documento de origen para citarlos como [n].",
        parameters={
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Pregunta autocontenida, sin depender del historial, "
                    "p. ej. 'fecha del primer parcial'",
                }
            },
            "required": ["consulta"],
        },
        run=run,
        read_only=True,
        origin="rag",
    )


async def _demo(question: str) -> None:
    from config import load_rag_settings
    from embeddings import Embedder
    from vector_store import VectorStore

    settings = load_rag_settings()
    store = VectorStore.load(settings.index_dir)
    # llm=None: la tool solo usa rag.retrieve(), que no llama al LLM (rag.answer() sí lo haría).
    rag = CourseRAG(None, Embedder(store.embedding_model), store, settings.top_k, settings.min_score)
    for rol, niveles in NIVELES_POR_ROL.items():
        tool = rag_tool(rag, niveles)
        print(f"== rol {rol}: {tool.name}({{'consulta': {question!r}}}) ->\n")
        print(await tool.run({"consulta": question}), "\n")


if __name__ == "__main__":
    from access import NIVELES_POR_ROL

    asyncio.run(_demo(" ".join(sys.argv[1:]) or "¿Cuándo es el simulacro del segundo parcial?"))
