"""Reto opcional: evaluación de la recuperación con hit rate@k.

Para cada pregunta de eval_preguntas.json se mira si el documento esperado aparece entre los
k primeros resultados. hit rate@k = preguntas acertadas / total. Cada tamaño de chunk se
indexa en memoria, así que no cambia el índice de index/. No usa el LLM ni necesita API key.

Uso:
    uv run python labs/02-rag/eval_retrieval.py                 # compara 80:20 y 30:5
    uv run python labs/02-rag/eval_retrieval.py 80:20 400:50    # tamaño:solapamiento
"""

import json
import sys

from config import LAB_DIR, load_rag_settings
from documents import chunk_documents, load_documents
from embeddings import Embedder
from vector_store import VectorStore

KS = (1, 2, 4)


def evaluar(store: VectorStore, embedder: Embedder, casos: list[dict]) -> dict[int, list[bool]]:
    aciertos = {k: [] for k in KS}
    for caso in casos:
        resultados = store.search(embedder.embed_query(caso["pregunta"]), max(KS))
        fuentes = [r.chunk.source for r in resultados]
        for k in KS:
            aciertos[k].append(caso["documento"] in fuentes[:k])
    return aciertos


def main() -> None:
    configs = [tuple(map(int, arg.split(":"))) for arg in sys.argv[1:]] or [(80, 20), (30, 5)]
    settings = load_rag_settings()
    casos = json.loads((LAB_DIR / "eval_preguntas.json").read_text(encoding="utf-8"))
    documentos = load_documents(settings.data_dir)
    embedder = Embedder(settings.embedding_model)

    print(f"{len(casos)} preguntas | modelo {settings.embedding_model}\n")
    print(f"{'chunk:solapamiento':>19} {'chunks':>7} " + " ".join(f"{'hit@' + str(k):>7}" for k in KS))
    detalle = {}
    for size, overlap in configs:
        chunks = chunk_documents(documentos, size, overlap)
        store = VectorStore(settings.embedding_model)
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
        aciertos = evaluar(store, embedder, casos)
        detalle[(size, overlap)] = aciertos
        tasas = " ".join(f"{sum(aciertos[k]) / len(casos):>7.0%}" for k in KS)
        print(f"{f'{size}:{overlap}':>19} {len(chunks):>7} {tasas}")

    k = max(KS)
    print(f"\nPreguntas falladas en el top {k}:")
    for (size, overlap), aciertos in detalle.items():
        fallas = [c["pregunta"] for c, ok in zip(casos, aciertos[k]) if not ok]
        print(f"  {size}:{overlap} -> {', '.join(fallas) if fallas else 'ninguna'}")


if __name__ == "__main__":
    main()
