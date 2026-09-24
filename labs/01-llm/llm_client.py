"""Cliente del LLM: la única parte de la aplicación que conoce al proveedor.

Recibe una lista de mensajes y parámetros; devuelve un LLMResponse.
El resto de la aplicación no importa `openai` ni sabe qué proveedor se usa.
"""

from dataclasses import dataclass

import openai

from config import Settings

# Un mensaje es un diccionario {"role": "system" | "user" | "assistant", "content": "..."}
Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str  # "stop" = terminó normalmente, "length" = se agotó max_tokens
    prompt_tokens: int
    completion_tokens: int


class LLMError(Exception):
    """Error al comunicarse con el proveedor, expresado sin detalles del SDK."""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = openai.OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        # TODO 1 (resuelto): llamada a Chat Completions. Sin temperature/max_tokens se usan los del .env.
        params = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        # Modelos de razonamiento: sin esto, el razonamiento interno puede agotar max_tokens.
        if self.settings.reasoning_effort:
            params["reasoning_effort"] = self.settings.reasoning_effort

        try:
            completion = self._client.chat.completions.create(**params)
        except openai.APIError as exc:
            # Se traduce a LLMError para que la aplicación no dependa del SDK.
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        # OpenRouter a veces responde HTTP 200 con un error y sin choices.
        if not completion.choices:
            detail = getattr(completion, "error", None) or "respuesta sin 'choices'"
            raise LLMError(f"El proveedor no devolvió una respuesta: {detail}")

        # TODO 2 (resuelto): respuesta del SDK -> LLMResponse.
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def create_client(settings: Settings):
    """Devuelve el cliente según LLM_PROVIDER. Con "fake" no se llama a ninguna API."""
    if settings.provider == "fake":
        from fake_llm_client import FakeLLMClient

        return FakeLLMClient(settings)
    return LLMClient(settings)


if __name__ == "__main__":
    # Prueba de humo: una sola llamada, sin interfaz ni historial.
    from config import load_settings

    client = create_client(load_settings())
    response = client.chat(
        [
            {"role": "system", "content": "Responde en una sola frase, en español."},
            {"role": "user", "content": "¿Qué es un Transformer en IA?"},
        ]
    )
    print(response)
