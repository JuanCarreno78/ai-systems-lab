"""Pruebas sin red ni API key gracias a FakeLLMClient.

Uso:
    uv run pytest labs/01-llm -v
"""

import json

import pytest
from pydantic import ValidationError

from config import load_settings
from fake_llm_client import FakeLLMClient
from llm_client import create_client
from prompts import SYSTEM_PROMPT, build_messages
from structured import QuestionAnalysis, analyze_question


# build_messages

def test_build_messages_sin_historial():
    messages = build_messages([], "¿Qué es la atención?")
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "¿Qué es la atención?"},
    ]


def test_build_messages_respeta_orden_system_historial_pregunta():
    history = [
        {"role": "user", "content": "Explica el positional encoding"},
        {"role": "assistant", "content": "Es una forma de codificar la posición..."},
    ]
    messages = build_messages(history, "Dame un ejemplo de eso")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == history
    assert messages[-1]["content"] == "Dame un ejemplo de eso"


def test_build_messages_no_modifica_el_historial():
    history = [{"role": "user", "content": "hola"}]
    build_messages(history, "otra pregunta")
    assert history == [{"role": "user", "content": "hola"}]


# analyze_question

def test_analyze_question_conocimiento_general():
    client = FakeLLMClient()
    analysis = analyze_question(client, "¿Qué es el mecanismo de atención?")
    assert isinstance(analysis, QuestionAnalysis)
    assert analysis.requiere_documentos_del_curso is False
    # structured.py debe pedir JSON y temperatura 0
    assert client.calls[0]["json_mode"] is True
    assert client.calls[0]["temperature"] == 0


def test_analyze_question_informacion_del_curso():
    analysis = analyze_question(FakeLLMClient(), "¿Qué temas entran en el parcial?")
    assert analysis.requiere_documentos_del_curso is True
    assert analysis.respuesta_corta == "No tengo esa información"


def test_analyze_question_json_invalido():
    client = FakeLLMClient(responses=["Claro, aquí va: {tema: atención"])
    with pytest.raises(json.JSONDecodeError):
        analyze_question(client, "¿Qué es la atención?")


def test_analyze_question_json_valido_pero_fuera_del_esquema():
    fuera_de_esquema = json.dumps(
        {"tema": "atención", "dificultad": "media", "requiere_documentos_del_curso": False, "respuesta_corta": "..."}
    )
    with pytest.raises(ValidationError):
        analyze_question(FakeLLMClient(responses=[fuera_de_esquema]), "¿Qué es la atención?")


# Selección del proveedor desde la configuración

def test_llm_provider_fake_se_elige_desde_el_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("LLM_MODEL", "fake-llm")
    client = create_client(load_settings())
    assert isinstance(client, FakeLLMClient)
    assert client.chat([{"role": "user", "content": "hola"}]).finish_reason == "stop"
