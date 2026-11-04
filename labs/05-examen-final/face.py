"""Reconocimiento facial con OpenCV: YuNet detecta el rostro y SFace lo convierte en un embedding.

- Detección: cv2.FaceDetectorYN da, por rostro, la caja y 5 landmarks (ojos, nariz y comisuras).
- Reconocimiento: cv2.FaceRecognizerSF alinea el rostro (alignCrop) y saca un vector de 128 números.
- Dos rostros son de la misma persona si la similitud coseno de sus vectores supera el umbral.

Los modelos son ONNX, corren en CPU y están en modelos/ (no se suben al repositorio).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import DETECTOR_MODEL, RECOGNIZER_MODEL, ROSTROS_DIR

LADO_DETECCION = 640  # lado mayor de la imagen con la que se detecta


@dataclass
class Rostro:
    fila: np.ndarray  # salida de YuNet: x, y, w, h, 10 coordenadas de landmarks y el puntaje

    @property
    def ancho(self) -> float:
        return float(self.fila[2])

    @property
    def giro(self) -> float:
        """Desplazamiento horizontal de la nariz respecto al punto medio de los ojos, dividido por la
        distancia entre los ojos. Cerca de 0 = de frente; el signo indica hacia qué lado gira."""
        ojo_1, ojo_2, nariz = self.fila[4:6], self.fila[6:8], self.fila[8:10]
        distancia_ojos = max(float(np.linalg.norm(ojo_1 - ojo_2)), 1.0)
        return float(nariz[0] - (ojo_1[0] + ojo_2[0]) / 2) / distancia_ojos


def leer_bytes(ruta: Path) -> np.ndarray:
    # OpenCV en Windows no abre rutas con caracteres como "ñ": el archivo se lee con Python.
    return np.fromfile(str(ruta), dtype=np.uint8)


def leer_imagen(ruta: Path) -> np.ndarray | None:
    return cv2.imdecode(leer_bytes(ruta), cv2.IMREAD_COLOR)


class FaceEngine:
    def __init__(self, detector: Path = DETECTOR_MODEL, recognizer: Path = RECOGNIZER_MODEL):
        for modelo in (detector, recognizer):
            if not modelo.exists():
                raise FileNotFoundError(f"Falta el modelo {modelo.name} en {modelo.parent}.")
        sin_config = np.array([], dtype=np.uint8)
        self._detector = cv2.FaceDetectorYN.create("onnx", leer_bytes(detector), sin_config, (320, 320), 0.8)
        self._recognizer = cv2.FaceRecognizerSF.create("onnx", leer_bytes(recognizer), sin_config)

    def detectar(self, imagen: np.ndarray) -> list[Rostro]:
        # YuNet no encuentra rostros muy grandes (selfies de cerca): se detecta en una copia reducida
        # y las coordenadas se devuelven al tamaño original, que es el que usa el reconocimiento.
        alto, ancho = imagen.shape[:2]
        escala = min(1.0, LADO_DETECCION / max(alto, ancho))
        reducida = cv2.resize(imagen, (round(ancho * escala), round(alto * escala))) if escala < 1 else imagen
        self._detector.setInputSize((reducida.shape[1], reducida.shape[0]))
        _, filas = self._detector.detect(reducida)
        if filas is None:
            return []
        filas = filas.copy()
        filas[:, :14] /= escala  # caja y landmarks; la última columna es el puntaje
        return [Rostro(fila) for fila in filas]

    def embedding(self, imagen: np.ndarray, rostro: Rostro) -> np.ndarray:
        alineado = self._recognizer.alignCrop(imagen, rostro.fila)
        vector = self._recognizer.feature(alineado).flatten().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def similitud(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos embeddings ya normalizados."""
    return float(np.dot(a, b))


def similitud_con_enrolado(vector: np.ndarray, enrolados: np.ndarray) -> float:
    """La mejor similitud entre un rostro y las fotos de enrolamiento de un usuario."""
    return float(np.max(enrolados @ vector))


def archivo_enrolamiento(user_id: str) -> Path:
    return ROSTROS_DIR / f"{user_id}.npy"


def cargar_enrolamiento(user_id: str) -> np.ndarray | None:
    archivo = archivo_enrolamiento(user_id)
    return np.load(archivo) if archivo.exists() else None
