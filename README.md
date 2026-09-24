# A2.3 Chatbot con LLM

Juan David Carreño Beltrán · Fundamentos de Inteligencia Artificial

Este repositorio tiene el código de la primera versión del Asistente Inteligente del Curso de IA,
desarrollado a partir del laboratorio `01-llm` del profesor
([ProfOmarPinzon/ai-systems-lab-students](https://github.com/ProfOmarPinzon/ai-systems-lab-students)).
El asistente es un chatbot de consola que recibe preguntas de los estudiantes y se las envía a un modelo
de lenguaje (LLM) por medio de una API.

El informe con el proceso, las pruebas y las respuestas a las preguntas de análisis está en
[INFORME.md](INFORME.md), y las salidas completas de cada prueba están en la carpeta [evidencias](evidencias/).

## Qué se hizo

- Se completaron los TODO 1 a 6 del código: la llamada al modelo, la construcción de los mensajes,
  el prompt de sistema, el historial de la conversación y la validación de la respuesta en JSON.
- Se hicieron las 8 pruebas del laboratorio. Las pruebas 1 a 7 se hicieron con un modelo local
  (Ollama con `qwen3:8b`) y en la prueba 8 se cambió a OpenRouter editando solo el archivo `.env`.
- Se probó el efecto de la temperatura y del límite de tokens en las respuestas.
- Como reto opcional se hizo un cliente falso (`LLM_PROVIDER=fake`) con 8 pruebas en `pytest`
  que funcionan sin internet.

El primer commit del repositorio es el código base del profesor sin cambios, así que la diferencia con
los commits siguientes muestra lo que se desarrolló.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `labs/01-llm/chatbot.py` | Programa de consola y conversación con el usuario |
| `labs/01-llm/llm_client.py` | Única parte que se comunica con el proveedor del modelo |
| `labs/01-llm/prompts.py` | Prompt de sistema y armado de los mensajes |
| `labs/01-llm/structured.py` | Pide la respuesta en JSON y la valida |
| `labs/01-llm/config.py` | Lee la configuración del archivo `.env` |
| `labs/01-llm/fake_llm_client.py` | Cliente falso del reto opcional |
| `labs/01-llm/test_fake_provider.py` | Pruebas del reto opcional |
| `labs/01-llm/README.md` | Enunciado original del laboratorio |

## Cómo ejecutarlo

Se usó Python 3.13 en Windows con `uv`, como indica el laboratorio. Desde la carpeta del repositorio:

```powershell
uv sync
Copy-Item .env.example .env
```

El `.env.example` ya tiene la configuración de Ollama que se usó en las pruebas 1 a 7. Para OpenRouter se
cambian los valores por los del bloque comentado y se pone la clave. Después:

```powershell
uv run python labs/01-llm/chatbot.py --debug
uv run python labs/01-llm/structured.py "¿Qué es el mecanismo de atención?"
uv run pytest labs/01-llm -v
```

El archivo `.env` tiene la clave de la API, por eso no está en el repositorio (está en `.gitignore`).
