"""Control de acceso: quién puede ver qué. Lo decide este código, no el LLM.

- La sesión trae el rol, el nombre y el código desde usuarios.json, nunca desde el chat.
- Cada rol ve ciertos niveles de documentos (RAG) y ciertas tools (MCP y locales).
- En las tools "mis_...", el código del estudiante lo pone el software desde la sesión: el modelo
  no ve ese parámetro, así que no puede pedir los datos de otro estudiante.
- Al ejecutar, se vuelve a revisar el rol (defensa en profundidad).
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from tools import Tool

NIVELES_POR_ROL = {
    "publico": {"publico"},
    "estudiante": {"publico", "estudiante"},
    "docente": {"publico", "estudiante", "docente"},
}

TOOLS_PUBLICAS = {"fecha_actual", "dias_hasta", "calcular_nota_final", "ver_calendario"}
TOOLS_POR_ROL = {
    "publico": TOOLS_PUBLICAS,
    "estudiante": TOOLS_PUBLICAS | {"consultar_mis_notas", "consultar_mis_solicitudes", "solicitar_supletorio"},
    "docente": TOOLS_PUBLICAS | {"consultar_notas", "consultar_solicitudes", "registrar_nota"},
}

# Tools cuyo código de estudiante se toma de la sesión y se oculta del modelo.
CODIGO_DESDE_SESION = {"consultar_mis_notas", "consultar_mis_solicitudes", "solicitar_supletorio"}


class AccesoDenegado(Exception):
    """El rol de la sesión no permite esa tool."""


@dataclass
class Sesion:
    telegram_user_id: int | None
    rol: str = "publico"
    user_id: str | None = None  # id en usuarios.json, p. ej. "integrante_1"
    nombre: str | None = None
    codigo: str | None = None
    expira: datetime | None = None
    historial: list[dict[str, Any]] = field(default_factory=list)  # una conversación por chat

    @property
    def autenticada(self) -> bool:
        return self.rol != "publico"

    def niveles(self) -> set[str]:
        return NIVELES_POR_ROL[self.rol]


def puede_usar(sesion: Sesion, tool_name: str) -> bool:
    return tool_name in TOOLS_POR_ROL[sesion.rol]


def tools_para(sesion: Sesion, todas: list[Tool]) -> list[Tool]:
    """Las tools que el modelo puede ver con esta sesión, ya adaptadas a la identidad."""
    disponibles = []
    for tool in todas:
        if not puede_usar(sesion, tool.name):
            continue
        if tool.name in CODIGO_DESDE_SESION:
            tool = con_codigo_de_sesion(tool, sesion)
        disponibles.append(tool)
    return disponibles


def con_codigo_de_sesion(tool: Tool, sesion: Sesion) -> Tool:
    """Quita `codigo` del esquema que ve el modelo y lo agrega al ejecutar, desde la sesión."""
    propiedades = {k: v for k, v in tool.parameters.get("properties", {}).items() if k != "codigo"}
    requeridos = [p for p in tool.parameters.get("required", []) if p != "codigo"]
    original = tool.run

    async def run(arguments: dict[str, Any]) -> str:
        if not sesion.codigo:
            raise AccesoDenegado("La sesión no tiene un código de estudiante.")
        # Cualquier "codigo" que intente mandar el modelo se reemplaza por el de la sesión.
        return await original({**arguments, "codigo": sesion.codigo})

    return replace(tool, parameters={**tool.parameters, "properties": propiedades, "required": requeridos}, run=run)
