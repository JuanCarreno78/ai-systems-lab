"""Enrolamiento local: guarda los embeddings del rostro de un integrante. No se hace desde el chat.

Recibe el id del usuario (de usuarios.json), su telegram_user_id y de 3 a 5 fotos. Cada foto debe
tener exactamente un rostro. Guarda solo los embeddings en rostros/<id>.npy (nunca las fotos) y
anota el telegram_user_id del usuario en usuarios.json.

Uso:
    uv run python labs/05-examen-final/enroll.py integrante_1 123456789 foto1.jpg foto2.jpg foto3.jpg
    uv run python labs/05-examen-final/enroll.py --borrar integrante_1     # borra sus embeddings
"""

import sys
from pathlib import Path

import numpy as np

from config import ROSTROS_DIR
from face import FaceEngine, archivo_enrolamiento, leer_imagen, similitud
from users import asignar_telegram, buscar_por_id


def embeddings_de_fotos(engine: FaceEngine, fotos: list[Path]) -> np.ndarray:
    vectores = []
    for foto in fotos:
        imagen = leer_imagen(foto)
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen {foto.name}.")
        rostros = engine.detectar(imagen)
        if len(rostros) != 1:
            raise ValueError(f"{foto.name} tiene {len(rostros)} rostros; debe tener exactamente uno.")
        vectores.append(engine.embedding(imagen, rostros[0]))
    return np.stack(vectores)


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--borrar":
        archivo = archivo_enrolamiento(args[1])
        archivo.unlink(missing_ok=True)
        print(f"Embeddings de {args[1]} borrados.")
        return
    if len(args) < 5 or len(args) > 7:
        sys.exit(__doc__)

    user_id, telegram_user_id, fotos = args[0], int(args[1]), [Path(f) for f in args[2:]]
    usuario = buscar_por_id(user_id)
    if usuario is None:
        sys.exit(f"No existe el usuario {user_id!r} en usuarios.json.")

    try:
        vectores = embeddings_de_fotos(FaceEngine(), fotos)
    except ValueError as exc:
        sys.exit(f"[enrolamiento] {exc}")

    # Las fotos deben ser de la misma persona: se revisa cada par antes de guardar.
    pares = [similitud(a, b) for i, a in enumerate(vectores) for b in vectores[i + 1 :]]
    ROSTROS_DIR.mkdir(exist_ok=True)
    np.save(archivo_enrolamiento(user_id), vectores)
    asignar_telegram(user_id, telegram_user_id)
    print(f"{usuario.nombre} ({usuario.rol}) enrolado con {len(vectores)} fotos.")
    print(f"Similitud entre sus fotos: mínima {min(pares):.3f}, máxima {max(pares):.3f}")
    print(f"Embeddings guardados en {archivo_enrolamiento(user_id)} (sin imágenes).")


if __name__ == "__main__":
    main()
