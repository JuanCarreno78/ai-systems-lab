"""Servidor MCP "Servicios académicos" (ficticio).

Expone, con el protocolo estándar MCP, datos y acciones del sistema académico:
  - tools (el MODELO decide pedirlas)
  - resources (la APLICACIÓN decide leerlos): academico://calendario

Es un programa independiente del asistente: no conoce al LLM, ni al proveedor, ni el
prompt. Cualquier cliente MCP (este asistente, un IDE, n8n, otro agente) puede usarlo.

Examen final: tools por rol. El servidor no sabe quién pregunta; el asistente decide qué tools
ve cada rol (access.py) y, en las tools "mis_...", pone el código desde la sesión. Por eso esas
tools sí reciben `codigo` aquí, pero el modelo nunca ve ese parámetro.
  - estudiante: consultar_mis_notas, consultar_mis_solicitudes, solicitar_supletorio
  - docente:    consultar_notas (con observaciones_docente), consultar_solicitudes, registrar_nota

Normalmente no se ejecuta a mano: mcp_client.py lo lanza como subproceso y le habla
por stdin/stdout (transporte stdio). Para exponerlo por HTTP (Streamable HTTP):
    uv run python labs/05-examen-final/mcp_server.py --http
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

DATA_FILE = Path(__file__).resolve().parent / "data" / "academico.json"
DATA = json.loads(DATA_FILE.read_text(encoding="utf-8"))
ESTUDIANTES: dict[str, Any] = DATA["estudiantes"]

# Solicitudes y notas nuevas de esta ejecución (en memoria: se pierden al cerrar el servidor).
SOLICITUDES: list[dict[str, Any]] = []

CodigoEstudiante = Annotated[str, Field(description="Código del estudiante, p. ej. 2026001")]
CodigoSesion = Annotated[str, Field(description="Código del estudiante autenticado (lo pone el sistema)")]

# IMPORTANTE: con el transporte stdio, stdout es el canal del protocolo. Un print() normal
# corrompería los mensajes; para depurar usa print(..., file=sys.stderr).
mcp = MCPServer(
    "servicios-academicos",
    instructions="Servicios académicos (ficticios) del curso de Inteligencia Artificial.",
    log_level="WARNING",  # los errores previstos (ToolError) no se imprimen en la consola del chat
)


def buscar_estudiante(codigo: str) -> dict[str, Any]:
    estudiante = ESTUDIANTES.get(codigo.strip())
    if estudiante is None:
        # ToolError = error previsto: el mensaje le llega al cliente (y al LLM) para que pueda corregir.
        raise ToolError(f"No existe un estudiante con código {codigo!r}.")
    return estudiante


def resumen_notas(codigo: str, estudiante: dict[str, Any]) -> dict[str, Any]:
    notas = estudiante["notas"]
    laboratorios = notas["laboratorios"]
    return {
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "notas": notas,
        "promedio_laboratorios": round(sum(laboratorios) / len(laboratorios), 2) if laboratorios else None,
    }


# --- Rol estudiante: solo sus propios datos ---------------------------------------------------


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_mis_notas(codigo: CodigoSesion) -> dict[str, Any]:
    """Consulta las notas del estudiante autenticado.

    Una nota null significa que aún no está registrada. Incluye el promedio de los laboratorios.
    No incluye observaciones del docente.
    """
    return resumen_notas(codigo, buscar_estudiante(codigo))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_mis_solicitudes(codigo: CodigoSesion) -> list[dict[str, Any]]:
    """Lista las solicitudes de supletorio del estudiante autenticado."""
    buscar_estudiante(codigo)
    return [solicitud for solicitud in SOLICITUDES if solicitud["codigo"] == codigo]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def solicitar_supletorio(
    codigo: CodigoSesion,
    parcial: Annotated[Literal[1, 2], Field(description="Número del parcial: 1 o 2")],
    causa: Annotated[
        Literal["incapacidad médica", "calamidad doméstica", "representación institucional"],
        Field(description="Causa justificada de la inasistencia"),
    ],
) -> dict[str, Any]:
    """Registra una solicitud de supletorio del estudiante autenticado para un parcial al que no asistió.

    Es una ACCIÓN: crea un registro en el sistema académico.
    """
    estudiante = buscar_estudiante(codigo)
    if any(s["codigo"] == codigo and s["parcial"] == parcial for s in SOLICITUDES):
        raise ToolError(f"Ya existe una solicitud de supletorio del parcial {parcial} para {codigo}.")
    solicitud = {
        "radicado": f"SUP-2026-{len(SOLICITUDES) + 1:03d}",
        "codigo": codigo,
        "nombre": estudiante["nombre"],
        "parcial": parcial,
        "causa": causa,
        "estado": "pendiente: adjuntar el soporte por Teams dentro de los tres días hábiles siguientes al parcial",
    }
    SOLICITUDES.append(solicitud)
    return solicitud


# --- Rol docente: cualquier estudiante --------------------------------------------------------


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_notas(codigo: CodigoEstudiante) -> dict[str, Any]:
    """Consulta las notas de cualquier estudiante del curso, con las observaciones del docente.

    Una nota null significa que aún no está registrada. Incluye el promedio de los laboratorios.
    """
    estudiante = buscar_estudiante(codigo)
    return {**resumen_notas(codigo, estudiante), "observaciones_docente": estudiante["observaciones_docente"]}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def consultar_solicitudes(codigo: CodigoEstudiante) -> list[dict[str, Any]]:
    """Lista las solicitudes de supletorio registradas por un estudiante."""
    buscar_estudiante(codigo)
    return [solicitud for solicitud in SOLICITUDES if solicitud["codigo"] == codigo]


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def registrar_nota(
    codigo: CodigoEstudiante,
    evaluacion: Annotated[
        Literal["parcial_1", "parcial_2", "proyecto"], Field(description="Evaluación a la que se le pone la nota")
    ],
    valor: Annotated[float, Field(ge=0.0, le=5.0, description="Nota de 0.0 a 5.0")],
) -> dict[str, Any]:
    """Registra o cambia la nota de un estudiante en una evaluación.

    Es una ACCIÓN: modifica el sistema académico.
    """
    estudiante = buscar_estudiante(codigo)
    anterior = estudiante["notas"][evaluacion]
    estudiante["notas"][evaluacion] = round(valor, 1)
    return {"codigo": codigo, "nombre": estudiante["nombre"], "evaluacion": evaluacion,
            "nota_anterior": anterior, "nota_nueva": estudiante["notas"][evaluacion]}


@mcp.resource("academico://calendario", description="Fechas clave del curso (parciales y proyecto final)")
def calendario() -> str:
    return "\n".join(f"{item['fecha']}: {item['evento']}" for item in DATA["calendario"])


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run("streamable-http")  # http://127.0.0.1:8000/mcp
    else:
        mcp.run()  # stdio: lo lanza el cliente
