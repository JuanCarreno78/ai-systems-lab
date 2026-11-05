"""Prueba de vida: revisa una video-nota frame por frame contra el reto pedido.

Pasa solo si se cumplen las tres verificaciones:
  1. hay exactamente un rostro en los frames analizados y es el mismo durante todo el video;
  2. ocurre el giro pedido, medido con los landmarks (nariz respecto al punto medio de los ojos);
  3. el frame más frontal supera el umbral contra el usuario enrolado (verificación 1:1).

El motivo exacto del rechazo solo va a la auditoría; al usuario no se le dice cuál falló.
Una foto plana mostrada en otra pantalla no cambia la posición de la nariz respecto a los ojos al
moverla o rotarla, por eso no pasa un reto de giro.

Uso (calibración con un video propio, sin Telegram):
    uv run python labs/05-examen-final/liveness.py <video.mp4> <reto> <id_usuario>
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from face import FaceEngine, similitud, similitud_con_enrolado

MUESTRAS = 24  # frames repartidos a lo largo del video
MIN_FRAMES_CON_ROSTRO = 0.6  # fracción mínima de frames con exactamente un rostro
# Cambio mínimo del giro (respecto al inicio) para contar un movimiento. En los videos de los integrantes
# los giros dieron entre 0.22 y 0.48, y una cara quieta se mueve menos de 0.10.
GIRO_MINIMO = 0.20
# En la imagen sin espejo, cuando la persona gira hacia SU izquierda la nariz se mueve hacia la derecha
# de la imagen y el giro aumenta. Si la cámara entrega el video en espejo, se invierte con -1.
SIGNO_IZQUIERDA = 1
ROTACIONES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class ResultadoVida:
    ok: bool
    motivo: str  # "ok" o la verificación que falló (solo para la auditoría)
    similitud: float | None = None
    detalle: dict = field(default_factory=dict)


def leer_frames(ruta: str, muestras: int = MUESTRAS) -> list[np.ndarray]:
    if not ruta.isascii():
        # OpenCV en Windows no abre rutas con caracteres como "ñ": se lee una copia temporal.
        with tempfile.TemporaryDirectory() as carpeta:
            copia = shutil.copy(ruta, Path(carpeta) / "video.mp4")
            return leer_frames(str(copia), muestras)
    video = cv2.VideoCapture(ruta)
    video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # aplica la rotación que trae el video del celular
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for indice in np.linspace(0, total - 1, min(muestras, total)).astype(int):
            video.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
            leido, frame = video.read()
            if leido:
                frames.append(frame)
    video.release()
    return frames


def rostro_derecho(rostro) -> bool:
    """Ojos arriba de la nariz y nariz arriba de la boca: la cara no está de lado ni al revés."""
    f = rostro.fila
    ojos_y, nariz_y, boca_y = (f[5] + f[7]) / 2, f[9], (f[11] + f[13]) / 2
    return bool(ojos_y < nariz_y < boca_y)


def orientar(frames: list[np.ndarray], engine: FaceEngine) -> tuple[list[np.ndarray], int]:
    """Algunos celulares guardan la video-nota girada. Se prueba con 0, 90, 180 y 270 grados en
    unos frames de muestra y se deja la rotación en la que la cara sale derecha con más confianza."""
    muestra = frames[:: max(1, len(frames) // 6)]
    mejor, mejor_puntaje = 0, 0.0
    for grados, rotacion in ROTACIONES.items():
        puntaje = 0.0
        for frame in muestra:
            girado = frame if rotacion is None else cv2.rotate(frame, rotacion)
            puntaje += sum(float(r.fila[14]) for r in engine.detectar(girado) if rostro_derecho(r))
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = grados, puntaje
    rotacion = ROTACIONES[mejor]
    if rotacion is None:
        return frames, 0
    return [cv2.rotate(f, rotacion) for f in frames], mejor


def movimiento_detectado(giros: list[float], reto: str) -> bool:
    """¿Los giros (en orden) muestran el movimiento del reto? Se mide respecto al inicio del video.
    Algunos celulares guardan la video-nota en espejo y desde el video no se puede saber, por eso se
    acepta el movimiento en los dos sentidos. Se sigue exigiendo el tipo de reto (un giro o los dos lados)."""
    base = float(np.median(giros[: max(2, len(giros) // 5)]))
    return any(
        _cumple_reto([(g - base) * signo for g in giros], reto) for signo in (SIGNO_IZQUIERDA, -SIGNO_IZQUIERDA)
    )


def _cumple_reto(relativos: list[float], reto: str) -> bool:
    # positivo = hacia la izquierda del usuario
    izquierda = [i for i, g in enumerate(relativos) if g >= GIRO_MINIMO]
    derecha = [i for i, g in enumerate(relativos) if g <= -GIRO_MINIMO]
    if reto == "izquierda":
        return bool(izquierda)
    if reto == "derecha":
        return bool(derecha)
    if reto == "izquierda_derecha":
        return bool(izquierda and derecha) and izquierda[0] < derecha[-1]
    if reto == "derecha_izquierda":
        return bool(izquierda and derecha) and derecha[0] < izquierda[-1]
    raise ValueError(f"Reto desconocido: {reto!r}")


def verificar_video(
    ruta: str, reto: str, engine: FaceEngine, enrolados: np.ndarray, umbral: float
) -> ResultadoVida:
    frames = leer_frames(ruta)
    if not frames:
        return ResultadoVida(False, "video_ilegible")
    frames, rotacion = orientar(frames, engine)

    analizados = []  # (frame, rostro) con exactamente un rostro
    for frame in frames:
        rostros = engine.detectar(frame)
        if len(rostros) > 1:
            return ResultadoVida(False, "varios_rostros", detalle={"frames": len(frames)})
        if len(rostros) == 1:
            analizados.append((frame, rostros[0]))
    alto, ancho = frames[0].shape[:2]
    detalle = {
        "frames": len(frames),
        "frames_con_rostro": len(analizados),
        "tamano": f"{ancho}x{alto}",
        "rotacion": rotacion,
    }
    if len(analizados) < MIN_FRAMES_CON_ROSTRO * len(frames):
        return ResultadoVida(False, "sin_rostro", detalle=detalle)

    # 1. El mismo rostro durante todo el video (contra el frame más frontal).
    frontal_frame, frontal = min(analizados, key=lambda par: abs(par[1].giro))
    vector_frontal = engine.embedding(frontal_frame, frontal)
    misma_persona = min(similitud(engine.embedding(f, r), vector_frontal) for f, r in analizados)
    detalle["misma_persona_min"] = round(misma_persona, 3)
    if misma_persona < umbral:
        return ResultadoVida(False, "cambio_de_rostro", detalle=detalle)

    # 2. El movimiento pedido.
    giros = [r.giro for _, r in analizados]
    detalle["giros"] = [round(g, 2) for g in giros]
    if not movimiento_detectado(giros, reto):
        return ResultadoVida(False, "movimiento_no_detectado", detalle=detalle)

    # 3. El frame frontal contra el usuario enrolado (1:1).
    valor = similitud_con_enrolado(vector_frontal, enrolados)
    if valor < umbral:
        return ResultadoVida(False, "rostro_no_coincide", round(valor, 3), detalle)
    return ResultadoVida(True, "ok", round(valor, 3), detalle)


if __name__ == "__main__":
    import sys

    from config import load_security_settings
    from face import cargar_enrolamiento

    ruta_video, tipo_reto, usuario = sys.argv[1], sys.argv[2], sys.argv[3]
    enrolados = cargar_enrolamiento(usuario)
    if enrolados is None:
        sys.exit(f"{usuario} no está enrolado. Ejecuta primero enroll.py.")
    resultado = verificar_video(
        ruta_video, tipo_reto, FaceEngine(), enrolados, load_security_settings().face_match_threshold
    )
    print(resultado)
