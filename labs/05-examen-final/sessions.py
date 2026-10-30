"""Sesiones, intentos de login y retos de prueba de vida, por telegram_user_id (en memoria).

- La sesión vence a los SESSION_MINUTES; al vencer o con /logout vuelve a "publico" y se borra
  el historial, porque podía tener datos restringidos.
- Después de LOGIN_MAX_ATTEMPTS fallos seguidos, la cuenta queda bloqueada un tiempo.
- Un reto es aleatorio, caduca a los LIVENESS_TIMEOUT_SECONDS y sirve una sola vez.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from access import Sesion
from config import SecuritySettings
from users import Usuario

# Reto -> instrucción que ve el usuario. Todos piden girar la cabeza (movimiento en 3D): una foto
# plana no lo puede imitar. "Acercar el rostro" se descartó porque una foto acercada a la cámara sí lo imita.
RETOS = {
    "izquierda": "gira la cabeza hacia TU izquierda y vuelve al frente",
    "derecha": "gira la cabeza hacia TU derecha y vuelve al frente",
    "izquierda_derecha": "gira la cabeza primero hacia TU izquierda y después hacia TU derecha",
    "derecha_izquierda": "gira la cabeza primero hacia TU derecha y después hacia TU izquierda",
}


@dataclass
class Reto:
    tipo: str
    emitido: datetime
    usado: bool = False


class SessionManager:
    def __init__(self, settings: SecuritySettings):
        self.settings = settings
        self._sesiones: dict[int, Sesion] = {}
        self._fallos: dict[int, int] = {}
        self._bloqueado_hasta: dict[int, datetime] = {}
        self._retos: dict[int, Reto] = {}

    # --- sesiones ---

    def obtener(self, telegram_user_id: int) -> tuple[Sesion, bool]:
        """La sesión del usuario y True si acaba de vencer."""
        sesion = self._sesiones.setdefault(telegram_user_id, Sesion(telegram_user_id))
        if sesion.autenticada and sesion.expira and datetime.now() >= sesion.expira:
            self.cerrar(telegram_user_id)
            return self._sesiones[telegram_user_id], True
        return sesion, False

    def iniciar(self, telegram_user_id: int, usuario: Usuario) -> Sesion:
        sesion = Sesion(
            telegram_user_id,
            rol=usuario.rol,
            user_id=usuario.id,
            nombre=usuario.nombre,
            codigo=usuario.codigo,
            expira=datetime.now() + timedelta(minutes=self.settings.session_minutes),
        )
        self._sesiones[telegram_user_id] = sesion
        self._fallos.pop(telegram_user_id, None)
        return sesion

    def cerrar(self, telegram_user_id: int) -> None:
        self._sesiones[telegram_user_id] = Sesion(telegram_user_id)

    # --- intentos y bloqueo ---

    def segundos_bloqueado(self, telegram_user_id: int) -> int:
        hasta = self._bloqueado_hasta.get(telegram_user_id)
        if hasta is None or datetime.now() >= hasta:
            self._bloqueado_hasta.pop(telegram_user_id, None)
            return 0
        return int((hasta - datetime.now()).total_seconds()) + 1

    def registrar_fallo(self, telegram_user_id: int) -> bool:
        """Suma un intento fallido. Devuelve True si con este fallo la cuenta quedó bloqueada."""
        fallos = self._fallos.get(telegram_user_id, 0) + 1
        if fallos >= self.settings.login_max_attempts:
            self._fallos.pop(telegram_user_id, None)
            self._bloqueado_hasta[telegram_user_id] = datetime.now() + timedelta(
                minutes=self.settings.login_lock_minutes
            )
            return True
        self._fallos[telegram_user_id] = fallos
        return False

    # --- retos de prueba de vida ---

    def emitir_reto(self, telegram_user_id: int) -> Reto:
        reto = Reto(secrets.choice(list(RETOS)), datetime.now())
        self._retos[telegram_user_id] = reto
        return reto

    def tomar_reto(self, telegram_user_id: int, enviado: datetime) -> tuple[Reto | None, str | None]:
        """El reto pendiente si sigue válido para un video enviado a esa hora; si no, el motivo."""
        reto = self._retos.get(telegram_user_id)
        if reto is None:
            return None, "sin_reto"
        if reto.usado:
            return None, "reto_usado"
        reto.usado = True  # un reto sirve una sola vez, pase o no la verificación
        if (enviado - reto.emitido).total_seconds() > self.settings.liveness_timeout_seconds:
            return None, "reto_vencido"
        return reto, None
