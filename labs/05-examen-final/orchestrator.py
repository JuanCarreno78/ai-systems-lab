"""Orquestador: compone LLM, RAG, tools locales y MCP en un solo asistente.

Dos estrategias para decidir qué capacidad usar:

  modo "router" (flujo fijo, decidido por el software después de clasificar):
      pregunta → router → "directa"   → LLM sin tools               (como el Lab 01)
                        → "documentos" → CourseRAG.answer           (como el Lab 02)
                        → "acciones"   → agente con tools y MCP     (como el Lab 03, con ciclo)

  modo "agente" (flujo dinámico, decidido por el modelo en cada paso):
      pregunta → agente con TODAS las capacidades como tools (incluida buscar_documentos)

El orquestador no sabe nada de la consola: recibe una pregunta y devuelve la respuesta
con su traza (ruta, pasos, fuentes, llamadas al LLM).

Examen final: cada respuesta se arma con la sesión. El RAG solo busca en los niveles del rol, el
agente solo recibe las tools del rol y las vuelve a revisar al ejecutarlas, y el aviso de
"necesitas iniciar sesión" lo escribe este código, no el LLM.
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Literal

from access import Sesion, puede_usar, tools_para
from agent import MAX_STEPS, Agent, AgentAnswer, AgentStep, ConfirmFn
from capabilities import rag_tool
from config import load_rag_settings, load_settings
from embeddings import Embedder
from llm_client import LLMClient, LLMResponse
from mcp_client import calendar_tool, load_mcp_tools, open_client
from prompts import build_direct_messages
from rag import CourseRAG
from router import RouterResult, route
from tools import LOCAL_TOOLS, Tool
from vector_store import VectorStore

Mode = Literal["router", "agente"]

PIDE_LOGIN = "Esa información está en documentos restringidos del curso. Para verla debes iniciar sesión con /login."
ROL_SIN_ACCESO = "Esa información es reservada y tu rol ({rol}) no tiene acceso a ella."
NOTA_PIDE_LOGIN = "Nota: hay más información sobre esto en documentos restringidos. Para verla debes iniciar sesión con /login."
NOTA_ROL_SIN_ACCESO = "Nota: hay información relacionada que es reservada, y tu rol ({rol}) no tiene acceso a ella."


@dataclass
class OrchestratorAnswer:
    text: str
    mode: Mode
    routing: RouterResult | None = None  # solo en modo router
    steps: list[AgentStep] = field(default_factory=list)  # traza del agente, si se usó
    sources: list[str] = field(default_factory=list)  # fuentes calculadas por el programa (ruta documentos)
    niveles_enviados: list[str] = field(default_factory=list)  # niveles de los fragmentos que vio el LLM
    responses: list[LLMResponse] = field(default_factory=list)  # TODAS las llamadas al LLM, en orden
    stopped_by_limit: bool = False
    bloqueado: str | None = None  # nivel restringido que tenía la respuesta, si el rol no lo puede ver

    def tools_used(self) -> list[str]:
        return [run.call.name for step in self.steps for run in step.tool_runs]

    def tools_denied(self) -> list[str]:
        return [run.call.name for step in self.steps for run in step.tool_runs if run.status == "denegada"]

    def tools_ok(self) -> list[str]:
        return [run.call.name for step in self.steps for run in step.tool_runs if run.status == "ok"]


def aviso_restringido(sesion: Sesion) -> str:
    return PIDE_LOGIN if not sesion.autenticada else ROL_SIN_ACCESO.format(rol=sesion.rol)


def nota_restringida(sesion: Sesion) -> str:
    return NOTA_PIDE_LOGIN if not sesion.autenticada else NOTA_ROL_SIN_ACCESO.format(rol=sesion.rol)


def describir_usuario(sesion: Sesion) -> str | None:
    """Nombre y rol para el prompt, tomados de la sesión (nunca del chat)."""
    return f"{sesion.nombre} (rol {sesion.rol})" if sesion.autenticada else None


class Orchestrator:
    def __init__(self, llm: LLMClient, rag: CourseRAG, tools: list[Tool], max_steps: int = MAX_STEPS):
        self.llm = llm
        self.rag = rag
        self.tools = tools  # todas las tools; cada respuesta usa solo las del rol
        self.max_steps = max_steps

    def _agent(self, sesion: Sesion, confirm: ConfirmFn, con_documentos: bool) -> Agent:
        tools = tools_para(sesion, self.tools)
        if con_documentos:
            tools = [rag_tool(self.rag, sesion.niveles()), *tools]

        def authorize(name: str) -> bool:
            return (name == "buscar_documentos" and con_documentos) or puede_usar(sesion, name)

        return Agent(self.llm, tools, confirm, self.max_steps, authorize, describir_usuario(sesion))

    async def answer(self, question: str, sesion: Sesion, confirm: ConfirmFn, mode: Mode = "router") -> OrchestratorAnswer:
        history = sesion.historial
        usuario = describir_usuario(sesion)
        if mode == "agente":
            agent = self._agent(sesion, confirm, con_documentos=True)
            return self._from_agent(await agent.run(question, history), mode)

        routing = route(self.llm, question)
        ruta = routing.decision.ruta
        if ruta == "directa":
            response = self.llm.chat(build_direct_messages(history, question, usuario))
            answer = OrchestratorAnswer(response.text, mode, responses=[response])
        elif ruta == "documentos":
            rag_answer = self.rag.answer(question, history, sesion.niveles(), usuario)
            if rag_answer.response is None:
                # Solo había respuesta en documentos que el rol no puede ver: responde el software.
                answer = OrchestratorAnswer(aviso_restringido(sesion), mode, bloqueado=rag_answer.bloqueado)
            else:
                texto = rag_answer.text
                if rag_answer.bloqueado:
                    texto += "\n\n" + nota_restringida(sesion)
                answer = OrchestratorAnswer(
                    texto,
                    mode,
                    sources=sorted({r.chunk.source for r in rag_answer.sources}),
                    niveles_enviados=sorted({r.chunk.nivel for r in rag_answer.sources}),
                    responses=[rag_answer.response],
                    bloqueado=rag_answer.bloqueado,
                )
        else:
            agent = self._agent(sesion, confirm, con_documentos=False)
            answer = self._from_agent(await agent.run(question, history), mode)
        answer.routing = routing
        answer.responses.insert(0, routing.response)
        return answer

    @staticmethod
    def _from_agent(result: AgentAnswer, mode: Mode) -> OrchestratorAnswer:
        return OrchestratorAnswer(
            result.text,
            mode,
            steps=result.steps,
            responses=[step.response for step in result.steps],
            stopped_by_limit=result.stopped_by_limit,
        )


async def create_orchestrator(stack: AsyncExitStack) -> Orchestrator:
    """Arma el sistema completo: configuración, LLM, índice, RAG, tools locales y servidor MCP.

    Lanza ValueError (configuración), FileNotFoundError (falta el índice) o un error de
    conexión con el servidor MCP. El servidor queda abierto hasta que se cierre `stack`.
    """
    settings = load_settings()
    rag_settings = load_rag_settings()
    store = VectorStore.load(rag_settings.index_dir)

    llm = LLMClient(settings)
    rag = CourseRAG(
        llm=llm,
        # La pregunta se convierte con el mismo modelo que indexó los documentos.
        embedder=Embedder(store.embedding_model),
        store=store,
        top_k=rag_settings.top_k,
        min_score=rag_settings.min_score,
    )
    mcp_client = await open_client(stack)
    tools = [*LOCAL_TOOLS, calendar_tool(mcp_client), *await load_mcp_tools(mcp_client)]
    return Orchestrator(llm, rag, tools)
