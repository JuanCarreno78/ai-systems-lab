# Actividad 02: Chatbot con RAG

Segunda versión del Asistente Inteligente del Curso de IA. Antes de preguntarle al modelo, el programa busca
en los documentos del curso (carpeta `data/`) los fragmentos que tienen que ver con la pregunta y se los manda
junto con ella. Así el asistente puede responder cosas del curso que el modelo no sabe, y decir de dónde las
sacó.

El enunciado lo entregó el profesor en PDF por Teams, y el código base está en su repositorio,
[ProfOmarPinzon/ai-systems-lab-students](https://github.com/ProfOmarPinzon/ai-systems-lab-students).

## Cómo está organizado

Hay dos partes que se corren en momentos distintos:

1. **Indexación** (`ingest.py`): lee los documentos, los parte en fragmentos, calcula el vector de cada uno y
   guarda todo en `index/`. Se corre una vez, y otra vez cuando cambian los documentos.
2. **Consulta** (`chatbot.py`): en cada pregunta busca los fragmentos más parecidos, arma el mensaje con ellos y
   se lo manda al modelo.

| Archivo | Qué hace |
|---|---|
| `data/` | Documentos del curso (ficticios): la fuente de la información |
| `documents.py` | Carga los documentos y los parte en fragmentos |
| `embeddings.py` | Convierte texto en vectores con un modelo local (fastembed) |
| `vector_store.py` | Guarda los vectores y busca los más parecidos a la pregunta |
| `ingest.py` | Crea el índice en `index/` (no usa el LLM) |
| `search.py` | Muestra qué fragmentos encuentra cada búsqueda, sin LLM |
| `rag.py` | Busca, arma el contexto y le pregunta al LLM |
| `prompts.py` | Prompt de sistema con las reglas de grounding |
| `chatbot.py` | Conversación en la consola |
| `eval_retrieval.py` | Reto opcional: mide el hit rate@k con `eval_preguntas.json` |

## Lo que se hizo

- Se completaron los TODO 1 a 6: chunking por palabras, búsqueda por similitud coseno, formato del contexto,
  mensajes con contexto, reglas de grounding y umbral de relevancia.
- Se agregó soporte para Ollama, igual que en el Lab 01, para correr el LLM en el computador (`qwen3:8b`).
- El umbral quedó en 0.40: las preguntas del curso dieron 0.49 o más y las que no son del curso 0.347 o menos.
- Se hicieron las 10 pruebas de la guía y los experimentos con el tamaño de chunk y `top_k`.
- Se agregó a `anuncios.md` un anuncio nuevo (cambio de horario de asesoría) para la prueba 9.
- Reto opcional: evaluación de la recuperación con hit rate@k para varios tamaños de chunk.

Las salidas completas de cada paso y de cada prueba están en la carpeta [evidencias](evidencias/).

## Cómo ejecutarlo

Desde la raíz del repositorio, con el `.env` configurado:

```powershell
uv sync
uv run python labs/02-rag/ingest.py
uv run python labs/02-rag/chatbot.py --debug
uv run python labs/02-rag/search.py "¿Cuándo es el primer parcial?"
uv run python labs/02-rag/eval_retrieval.py 80:20 30:5
```

La carpeta `index/` se genera con `ingest.py`, por eso no está en el repositorio.
