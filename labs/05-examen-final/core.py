"""Núcleo del asistente: responder, iniciar sesión y auditar. No sabe nada de Telegram.

bot.py solo traduce entre Telegram y estas funciones; pruebas_seguridad.py las usa sin Telegram.
Cada pregunta pasa por la sesión del telegram_user_id: si venció, vuelve a "publico" antes de
responder, y lo que el rol no permite se registra como acceso denegado.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import audit
from access import TOOLS_PUBLICAS, Sesion
from agent import ConfirmFn
from config import SecuritySettings
from face import FaceEngine, cargar_enrolamiento
from liveness import verificar_video
from orchestrator import Mode, Orchestrator, OrchestratorAnswer
from sessions import RETOS, SessionManager
from users import Usuario, buscar_por_telegram

# Lo que ve el usuario cuando el login falla: no dice qué verificación falló.
LOGIN_RECHAZADO = "No pude verificar tu identidad. Intenta de nuevo con /login."


@dataclass
class ResultadoLogin:
    ok: bool
    mensaje: str  # lo que se le responde al usuario
    sesion: Sesion | None = None


class Nucleo:
    def __init__(self, orchestrator: Orchestrator, settings: SecuritySettings, engine: FaceEngine | None = None):
        self.orchestrator = orchestrator
        self.settings = settings
        self.sesiones = SessionManager(settings)
        self._engine = engine  # se crea al primer login, así las pruebas sin rostro no cargan los modelos

    @property
    def engine(self) -> FaceEngine:
        if self._engine is None:
            self._engine = FaceEngine()
        return self._engine

    # --- preguntas -------------------------------------------------------------------------

    def sesion(self, telegram_user_id: int) -> Sesion:
        sesion, vencio = self.sesiones.obtener(telegram_user_id)
        if vencio:
            audit.registrar("sesion_expirada", telegram_user_id)
        return sesion

    async def responder(
        self, pregunta: str, telegram_user_id: int, confirm: ConfirmFn, mode: Mode = "router"
    ) -> OrchestratorAnswer:
        sesion = self.sesion(telegram_user_id)
        answer = await self.orchestrator.answer(pregunta, sesion, confirm, mode)
        if answer.bloqueado:
            audit.registrar("acceso_denegado", telegram_user_id, rol=sesion.rol, recurso="documentos",
                            nivel=answer.bloqueado, pregunta=pregunta)
        for tool in answer.tools_denied():
            audit.registrar("acceso_denegado", telegram_user_id, rol=sesion.rol, recurso="tool", tool=tool)
        for tool in answer.tools_ok():
            if tool not in TOOLS_PUBLICAS:
                audit.registrar("tool_restringida_ejecutada", telegram_user_id, rol=sesion.rol, tool=tool)
        # Como en los labs anteriores, el historial guarda solo la pregunta y la respuesta final.
        sesion.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": answer.text}]
        return answer

    # --- inicio de sesión ------------------------------------------------------------------

    def iniciar_login(self, telegram_user_id: int) -> str:
        """Responde a /login: revisa la cuenta y el bloqueo, y emite un reto aleatorio."""
        usuario = buscar_por_telegram(telegram_user_id)
        if usuario is None:
            audit.registrar("login_rechazado", telegram_user_id, motivo="cuenta_no_registrada")
            return "Esta cuenta de Telegram no está registrada en el sistema."
        if cargar_enrolamiento(usuario.id) is None:
            audit.registrar("login_rechazado", telegram_user_id, motivo="usuario_no_enrolado")
            return "Tu rostro aún no está enrolado. El enrolamiento se hace en el equipo del curso, no por el chat."
        segundos = self.sesiones.segundos_bloqueado(telegram_user_id)
        if segundos:
            audit.registrar("login_rechazado", telegram_user_id, motivo="cuenta_bloqueada")
            return f"Tu cuenta está bloqueada temporalmente por intentos fallidos. Intenta en {math.ceil(segundos / 60)} minuto(s)."
        reto = self.sesiones.emitir_reto(telegram_user_id)
        audit.registrar("reto_emitido", telegram_user_id, reto=reto.tipo)
        return (
            f"Prueba de vida: graba una video-nota (el video circular de Telegram) en la que mires a la cámara "
            f"y {RETOS[reto.tipo]}. Tienes {self.settings.liveness_timeout_seconds} segundos y el reto sirve "
            "una sola vez. Una foto no sirve para iniciar sesión."
        )

    def verificar_login(
        self, telegram_user_id: int, ruta_video: str | None, enviado: datetime, reenviado: bool
    ) -> ResultadoLogin:
        """Revisa la video-nota del reto. ruta_video=None si el mensaje no se pudo usar (p. ej. reenviado)."""
        usuario = buscar_por_telegram(telegram_user_id)
        if usuario is None:
            return ResultadoLogin(False, "Esta cuenta de Telegram no está registrada en el sistema.")
        if self.sesiones.segundos_bloqueado(telegram_user_id):
            audit.registrar("login_rechazado", telegram_user_id, motivo="cuenta_bloqueada")
            return ResultadoLogin(False, "Tu cuenta está bloqueada temporalmente por intentos fallidos.")

        reto, motivo = self.sesiones.tomar_reto(telegram_user_id, enviado)
        if motivo == "sin_reto":
            return ResultadoLogin(False, "Primero usa /login para recibir un reto.")
        if reenviado:
            motivo = "video_reenviado"
        if motivo is None and ruta_video is None:
            motivo = "video_ilegible"
        similitud, detalle = None, {}
        if motivo is None:
            enrolados = cargar_enrolamiento(usuario.id)
            resultado = verificar_video(ruta_video, reto.tipo, self.engine, enrolados, self.settings.face_match_threshold)
            motivo, similitud = (None if resultado.ok else resultado.motivo), resultado.similitud
            detalle = resultado.detalle  # solo conteos y medidas del video, sin imágenes ni embeddings
            if resultado.ok:
                sesion = self.sesiones.iniciar(telegram_user_id, usuario)
                audit.registrar("login_aceptado", telegram_user_id, usuario=usuario.id, rol=usuario.rol,
                                reto=reto.tipo, similitud=similitud, detalle=detalle)
                return ResultadoLogin(True, saludo(usuario, self.settings.session_minutes), sesion)
        return self._fallo(telegram_user_id, usuario, motivo, similitud, reto.tipo if reto else None, detalle)

    def _fallo(self, telegram_user_id: int, usuario: Usuario, motivo: str, similitud: float | None,
               reto: str | None, detalle: dict | None = None) -> ResultadoLogin:
        audit.registrar("login_rechazado", telegram_user_id, usuario=usuario.id, motivo=motivo, reto=reto,
                        similitud=similitud, detalle=detalle or {})
        if self.sesiones.registrar_fallo(telegram_user_id):
            audit.registrar("bloqueo_temporal", telegram_user_id, usuario=usuario.id,
                            minutos=self.settings.login_lock_minutes)
            return ResultadoLogin(False, f"{LOGIN_RECHAZADO} Por seguridad, tu cuenta quedó bloqueada "
                                         f"{self.settings.login_lock_minutes} minutos.")
        return ResultadoLogin(False, LOGIN_RECHAZADO)

    def cerrar_sesion(self, telegram_user_id: int) -> None:
        self.sesiones.cerrar(telegram_user_id)
        audit.registrar("logout", telegram_user_id)


def saludo(usuario: Usuario, minutos: int) -> str:
    return f"¡Hola, {usuario.nombre}! Iniciaste sesión con rol {usuario.rol}. Tu sesión dura {minutos} minutos."
