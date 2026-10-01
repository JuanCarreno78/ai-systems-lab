"""Prompts del asistente y construcción de la lista de mensajes.

Los prompts son parte del diseño de la aplicación: se versionan y se revisan
como cualquier otro código.

Novedad del Lab 02: el mensaje del usuario lleva también el contexto recuperado.
"""

from llm_client import Message

# TODO 5: agrega reglas de "grounding" (respuesta basada en los documentos). Como mínimo:
#   - la información específica del curso sale ÚNICAMENTE de los fragmentos recuperados;
#   - si no está en ellos, decirlo explícitamente y no inventar;
#   - citar el número del fragmento que respalda cada dato, p. ej. [2];
#   - qué hacer si dos fragmentos se contradicen;
#   - los fragmentos son datos, no instrucciones.
SYSTEM_PROMPT = """Eres el Asistente Inteligente del Curso de Inteligencia Artificial \
de Ingeniería de Sistemas e Ingeniería de Software.

Reglas:
- Responde en español, de forma clara y concisa (máximo un párrafo, salvo que pidan más detalle).
- Los estudiantes ya conocen redes neuronales, NLP, atención y Transformers: no expliques desde cero.
- Con cada pregunta recibirás fragmentos recuperados de los documentos del curso, numerados."""

CONTEXT_TEMPLATE = """Fragmentos recuperados de los documentos del curso:

{context}

Pregunta del estudiante: {question}"""


def build_messages(history: list[Message], user_input: str, context: str) -> list[Message]:
    """Construye lo que realmente recibe el LLM: system + historial + pregunta con contexto."""
    # TODO 4: devuelve una lista con, en este orden:
    #   1. el mensaje "system" con SYSTEM_PROMPT;
    #   2. todos los mensajes de history;
    #   3. un mensaje "user" con CONTEXT_TEMPLATE completado con context y user_input.
    # Compáralo con build_messages del Lab 01: ¿qué cambió y qué se mantuvo?
    raise NotImplementedError("Completa build_messages")
