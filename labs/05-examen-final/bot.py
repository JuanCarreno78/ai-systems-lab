"""Adaptador de Telegram: traduce entre Telegram y el núcleo (core.py). No tiene lógica de negocio.

Usa long polling: el bot le pregunta a Telegram si hay mensajes nuevos, así que corre desde un
portátil sin URL pública. Cada usuario tiene su propia sesión e historial (por telegram_user_id).
La confirmación de acciones es la ConfirmFn del Lab 04, pero con botones de Telegram.

Uso (requiere TELEGRAM_BOT_TOKEN en el .env, el índice de ingest.py y los modelos en modelos/):
    uv run python labs/05-examen-final/bot.py
"""

import asyncio
import logging
import tempfile
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from agent import ConfirmFn
from config import load_security_settings, load_telegram_token
from core import Nucleo
from llm_client import LLMError
from orchestrator import create_orchestrator
from prompts import DESARROLLADOR
from tools import Tool
from users import buscar_por_telegram

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # no imprimir cada consulta a Telegram
log = logging.getLogger("bot")

CONFIRMACION_SEGUNDOS = 60
AYUDA = (
    "Puedes preguntarme por el curso en texto. Lo público lo respondo a cualquiera; la información "
    "restringida y tus datos académicos necesitan iniciar sesión.\n\n"
    "/start  presentación\n"
    "/quiensoy  tu telegram_user_id y el estado de tu sesión\n"
    "/login  iniciar sesión con tu rostro y una prueba de vida (video-nota)\n"
    "/logout  cerrar sesión\n"
    "/ayuda  esta ayuda"
)


def nucleo(context: ContextTypes.DEFAULT_TYPE) -> Nucleo:
    return context.application.bot_data["nucleo"]


# --- comandos ---------------------------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sesion = nucleo(context).sesion(update.effective_user.id)
    saludo = f"¡Hola, {sesion.nombre}!" if sesion.autenticada else "¡Hola!"
    await update.message.reply_text(
        f"{saludo} Soy el Asistente del Curso de IA desarrollado por {DESARROLLADOR}.\n\n{AYUDA}"
    )


async def quiensoy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    sesion = nucleo(context).sesion(user_id)
    usuario = buscar_por_telegram(user_id)
    registro = f"registrado como {usuario.nombre} (rol {usuario.rol})" if usuario else "no registrado"
    estado = (
        f"sesión activa como {sesion.nombre}, rol {sesion.rol}, vence a las {sesion.expira:%H:%M}"
        if sesion.autenticada
        else "sin sesión (acceso público)"
    )
    await update.message.reply_text(f"telegram_user_id: {user_id}\nCuenta: {registro}\nEstado: {estado}")


async def login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(nucleo(context).iniciar_login(update.effective_user.id))


async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    nucleo(context).cerrar_sesion(update.effective_user.id)
    await update.message.reply_text("Sesión cerrada. Ahora solo tienes acceso a la información pública.")


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(AYUDA)


# --- preguntas y confirmación con botones -----------------------------------------------------


def confirmar_con_botones(update: Update, context: ContextTypes.DEFAULT_TYPE) -> ConfirmFn:
    pendientes: dict[str, Any] = context.application.bot_data["pendientes"]

    async def confirm(tool: Tool, arguments: dict[str, Any]) -> bool:
        clave = uuid.uuid4().hex[:12]
        respuesta = asyncio.get_running_loop().create_future()
        pendientes[clave] = (respuesta, update.effective_user.id)
        botones = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Sí", callback_data=f"si:{clave}"), InlineKeyboardButton("No", callback_data=f"no:{clave}")]]
        )
        argumentos = ", ".join(f"{k}={v}" for k, v in arguments.items())
        await update.effective_chat.send_message(
            f"El asistente quiere ejecutar una acción:\n{tool.name}({argumentos})\n¿La autorizas?", reply_markup=botones
        )
        try:
            return await asyncio.wait_for(respuesta, CONFIRMACION_SEGUNDOS)
        except TimeoutError:
            return False  # sin respuesta, la acción no se ejecuta
        finally:
            pendientes.pop(clave, None)

    return confirm


async def boton(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    decision, _, clave = query.data.partition(":")
    pendiente = context.application.bot_data["pendientes"].get(clave)
    # Solo cuenta el botón de la misma persona que hizo la pregunta.
    if pendiente is None or pendiente[1] != query.from_user.id:
        await query.answer("Esta confirmación ya no es válida.")
        return
    respuesta, _ = pendiente
    if not respuesta.done():
        respuesta.set_result(decision == "si")
    await query.answer()
    await query.edit_message_text(f"{query.message.text}\n\nRespuesta: {'Sí' if decision == 'si' else 'No'}")


async def texto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await update.effective_chat.send_action("typing")
    try:
        answer = await nucleo(context).responder(update.message.text, user_id, confirmar_con_botones(update, context))
    except LLMError as exc:
        log.error("Error del LLM para %s: %s", user_id, exc)
        await update.message.reply_text("No pude responder en este momento por un problema con el modelo. Intenta de nuevo.")
        return
    except Exception:  # p. ej. el servidor MCP no responde: el bot sigue funcionando
        log.exception("Error al responder a %s", user_id)
        await update.message.reply_text("Ocurrió un error al procesar tu pregunta. Intenta de nuevo en un momento.")
        return

    # Traza del orquestador (como --debug del Lab 04): qué niveles de documentos y qué tools se usaron.
    ruta = answer.routing.decision.ruta if answer.routing else "agente"
    traza = [f"{r.call.name}:{r.status}" for s in answer.steps for r in s.tool_runs]
    log.info("traza user=%s rol=%s ruta=%s niveles_enviados=%s tools=%s bloqueado=%s",
             user_id, nucleo(context).sesion(user_id).rol, ruta, answer.niveles_enviados, traza, answer.bloqueado)
    respuesta = answer.text or "(sin respuesta)"
    if answer.sources:
        respuesta += f"\n\nFuentes: {', '.join(answer.sources)}"
    await update.message.reply_text(respuesta)


# --- prueba de vida ---------------------------------------------------------------------------


async def video_nota(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mensaje = update.message
    reenviado = mensaje.forward_origin is not None
    enviado = mensaje.date.astimezone().replace(tzinfo=None)  # hora local, como la del reto
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = None
        if not reenviado:
            ruta = str(Path(carpeta) / "video_nota.mp4")
            archivo = await mensaje.video_note.get_file()
            await archivo.download_to_drive(ruta)
        # El análisis del video usa la CPU: se hace en otro hilo para no detener el bot.
        resultado = await asyncio.to_thread(nucleo(context).verificar_login, update.effective_user.id, ruta, enviado, reenviado)
    # Al salir del bloque se borra el video: no se guarda ninguna imagen del rostro.
    await mensaje.reply_text(resultado.mensaje)


async def foto_o_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Una foto o un video normal no sirve para iniciar sesión. Usa /login y responde con una video-nota "
        "(mantén presionado el botón de la cámara en el chat)."
    )


# --- arranque ---------------------------------------------------------------------------------


async def al_iniciar(app: Application) -> None:
    stack = AsyncExitStack()
    await stack.__aenter__()
    app.bot_data["stack"] = stack
    app.bot_data["nucleo"] = Nucleo(await create_orchestrator(stack), load_security_settings())
    app.bot_data["pendientes"] = {}
    log.info("Asistente listo: núcleo, índice y servidor MCP iniciados")


async def al_cerrar(app: Application) -> None:
    await app.bot_data["stack"].aclose()


async def error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Error no manejado: %s", context.error, exc_info=context.error)


def main() -> None:
    try:
        token = load_telegram_token()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return
    app = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)  # el botón de confirmación llega mientras se espera la respuesta
        .post_init(al_iniciar)
        .post_shutdown(al_cerrar)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("quiensoy", quiensoy))
    app.add_handler(CommandHandler("login", login))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("ayuda", ayuda))
    app.add_handler(CallbackQueryHandler(boton))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, video_nota))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, foto_o_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, texto))
    app.add_error_handler(error)
    log.info("Iniciando el bot (long polling). Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
