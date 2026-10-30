"""Tabla FAR/FRR para elegir el umbral del reconocimiento facial.

Lee una carpeta con una subcarpeta por persona (fotos .jpg o .png):
  - pares genuinos: dos fotos de la misma persona;
  - pares impostores: dos fotos de personas distintas.
Para cada umbral:
  - FAR (falsa aceptación) = impostores con similitud >= umbral / total de impostores
  - FRR (falso rechazo)    = genuinos con similitud < umbral / total de genuinos

Las fotos van en rostros/ (no se suben al repositorio) y se borran al terminar.

Uso:
    uv run python labs/05-examen-final/umbral.py labs/05-examen-final/rostros/fotos_umbral
"""

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from face import FaceEngine, similitud

UMBRALES = [0.25, 0.30, 0.363, 0.40, 0.45, 0.50, 0.55, 0.60]


def cargar(engine: FaceEngine, carpeta: Path) -> dict[str, list[np.ndarray]]:
    personas: dict[str, list[np.ndarray]] = {}
    for sub in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        for foto in sorted(sub.iterdir()):
            if foto.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            imagen = cv2.imread(str(foto))
            rostros = engine.detectar(imagen) if imagen is not None else []
            if len(rostros) != 1:
                print(f"  (se omite {sub.name}/{foto.name}: {len(rostros)} rostros)")
                continue
            personas.setdefault(sub.name, []).append(engine.embedding(imagen, rostros[0]))
    return personas


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    personas = cargar(FaceEngine(), Path(sys.argv[1]))
    genuinos = [similitud(a, b) for vs in personas.values() for a, b in combinations(vs, 2)]
    impostores = [
        similitud(a, b)
        for (_, va), (_, vb) in combinations(personas.items(), 2)
        for a in va
        for b in vb
    ]
    print(f"\nPersonas: {len(personas)} ({', '.join(f'{k}: {len(v)} fotos' for k, v in personas.items())})")
    print(f"Pares genuinos: {len(genuinos)}   Pares impostores: {len(impostores)}")
    if len(genuinos) < 10 or len(impostores) < 20:
        print("AVISO: el examen pide al menos 10 pares genuinos y 20 impostores.")
    print(f"Genuinos:   mínima {min(genuinos):.3f}  promedio {np.mean(genuinos):.3f}  máxima {max(genuinos):.3f}")
    print(f"Impostores: mínima {min(impostores):.3f}  promedio {np.mean(impostores):.3f}  máxima {max(impostores):.3f}\n")

    print(f"{'umbral':>7} {'FAR':>7} {'FRR':>7}  impostores aceptados / genuinos rechazados")
    for umbral in UMBRALES:
        aceptados = sum(s >= umbral for s in impostores)
        rechazados = sum(s < umbral for s in genuinos)
        print(f"{umbral:>7.3f} {aceptados / len(impostores):>7.1%} {rechazados / len(genuinos):>7.1%}  "
              f"{aceptados}/{len(impostores)} / {rechazados}/{len(genuinos)}")


if __name__ == "__main__":
    main()
