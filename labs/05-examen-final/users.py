"""Usuarios del sistema (data/usuarios.json): nombre, rol, código y telegram_user_id.

Es la única fuente del rol y del código de una persona. Nada de esto sale del chat.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from config import USUARIOS_FILE


@dataclass(frozen=True)
class Usuario:
    id: str
    nombre: str
    rol: str
    codigo: str | None
    telegram_user_id: int | None


def cargar_usuarios(archivo: Path = USUARIOS_FILE) -> list[Usuario]:
    data = json.loads(archivo.read_text(encoding="utf-8"))
    return [
        Usuario(u["id"], u["nombre"], u["rol"], u.get("codigo_estudiante"), u.get("telegram_user_id"))
        for u in data["usuarios"]
    ]


def buscar_por_telegram(telegram_user_id: int, archivo: Path = USUARIOS_FILE) -> Usuario | None:
    return next((u for u in cargar_usuarios(archivo) if u.telegram_user_id == telegram_user_id), None)


def buscar_por_id(user_id: str, archivo: Path = USUARIOS_FILE) -> Usuario | None:
    return next((u for u in cargar_usuarios(archivo) if u.id == user_id), None)


def asignar_telegram(user_id: str, telegram_user_id: int, archivo: Path = USUARIOS_FILE) -> None:
    """Guarda el telegram_user_id de un usuario (lo usa enroll.py)."""
    data = json.loads(archivo.read_text(encoding="utf-8"))
    for usuario in data["usuarios"]:
        if usuario["id"] == user_id:
            usuario["telegram_user_id"] = telegram_user_id
            break
    else:
        raise ValueError(f"No existe el usuario {user_id!r} en {archivo.name}.")
    archivo.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
