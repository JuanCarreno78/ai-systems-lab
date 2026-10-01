# Asistente Inteligente del Curso de IA

Juan David Carreño Beltran, Fundamentos de Inteligencia Artificial

Este repositorio tiene los laboratorios de la unidad 5 del curso, desarrollados a partir del repositorio del
profesor ([ProfOmarPinzon/ai-systems-lab-students](https://github.com/ProfOmarPinzon/ai-systems-lab-students)).
Cada laboratorio construye una versión nueva del Asistente Inteligente del Curso de IA.

| Lab | Carpeta | Qué hace el asistente |
|---|---|---|
| 01 | [`labs/01-llm`](labs/01-llm/README.md) | Le manda la pregunta directo al modelo de lenguaje (LLM) |
| 02 | [`labs/02-rag`](labs/02-rag/README.md) | Primero busca en los documentos del curso y le manda al modelo lo que encontró (RAG) |

Cada carpeta tiene su README con lo que se hizo, y su carpeta de evidencias con las salidas de las pruebas.
El informe del Lab 01 también está en [INFORME.md](INFORME.md). Los commits que dicen "Código base" son el
código del profesor sin cambios, así que la diferencia con los commits siguientes muestra lo que se desarrolló.

## Cómo ejecutarlo

Se usó Python 3.13 en Windows con `uv`. Desde la carpeta del repositorio:

```powershell
uv sync
Copy-Item .env.example .env
```

El `.env.example` ya tiene la configuración de Ollama (modelo local `qwen3:8b`) que se usó en las pruebas, y
un bloque comentado para OpenRouter. Después:

```powershell
uv run python labs/01-llm/chatbot.py --debug
uv run pytest labs/01-llm -v

uv run python labs/02-rag/ingest.py
uv run python labs/02-rag/chatbot.py --debug
```

El archivo `.env` tiene la clave de la API y la carpeta `labs/02-rag/index/` se genera con `ingest.py`, por eso
ninguno de los dos está en el repositorio (están en `.gitignore`).

`pyproject.toml` limita la versión de `mmh3` (una dependencia de fastembed) a menos de 5.3, porque en este
equipo el Control de aplicaciones de Windows bloqueaba la versión 5.3.1.
