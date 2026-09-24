"""Proveedor falso (reto opcional): mismo método chat que LLMClient, sin llamar a ninguna API.

Se activa con LLM_PROVIDER=fake. Se usa en las pruebas de test_fake_provider.py.
"""

import json

from config import Settings
from llm_client import LLMResponse, Message

# Palabras que indican que la pregunta depende de información propia del curso.
COURSE_KEYWORDS = ("parcial", "examen", "nota", "fecha", "entrega", "programa", "cronograma", "curso")


class FakeLLMClient:
    def __init__(self, settings: Settings | None = None, responses: list[str] | None = None):
        self.settings = settings
        self._responses = list(responses or [])  # respuestas fijas, en orden
        self.calls: list[dict] = []  # peticiones recibidas, para revisarlas en las pruebas

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        )
        if self._responses:
            text = self._responses.pop(0)
        elif json_mode:
            text = self._analysis(messages[-1]["content"])
        else:
            text = f"[fake] Recibí {len(messages)} mensajes. Última pregunta: {messages[-1]['content']}"

        return LLMResponse(
            text=text,
            model="fake-llm",
            finish_reason="stop",
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
            completion_tokens=len(text.split()),
        )

    @staticmethod
    def _analysis(question: str) -> str:
        from_course = any(word in question.lower() for word in COURSE_KEYWORDS)
        return json.dumps(
            {
                "tema": question.strip("¿?").strip()[:40],
                "dificultad": "basica",
                "requiere_documentos_del_curso": from_course,
                "respuesta_corta": "No tengo esa información" if from_course else "Respuesta de prueba.",
            },
            ensure_ascii=False,
        )
