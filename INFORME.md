# A2.3. Informe chatbot con LLM

Nombre: Juan David Carreño Beltran  
Código: 000550919  
Asignatura: Fundamentos de Inteligencia Artificial  
Docente: Omar Pinzón  
Fecha: 29 de septiembre de 2026  
Repositorio: https://github.com/JuanCarreno78/ai-systems-lab  

## 1. Introducción

Este informe presenta la primera versión del Asistente Inteligente del Curso de IA: un chatbot de consola que envía las preguntas del estudiante a un modelo de lenguaje (LLM) por medio de una API. El objetivo fue usar el modelo como una pieza más del programa: enviarle mensajes, controlar su respuesta con parámetros, separar el proveedor del resto del código y validar lo que devuelve.

El profesor entregó el código base con seis partes por completar (TODO). Cada archivo tiene una tarea: config.py lee el .env, llm_client.py habla con el proveedor, prompts.py arma los mensajes, chatbot.py maneja la conversación y structured.py pide y valida respuestas en JSON. Gracias a esa separación se pudo cambiar de proveedor sin tocar la lógica del chatbot.

Se trabajó en Windows 11 con Python 3.13 y Visual Studio Code. Las pruebas 1 a 7 se hicieron con Ollama (un programa que ejecuta modelos en el propio computador) y el modelo qwen3:8b, sin API key ni costo. La prueba 8 se hizo con OpenRouter, un servicio en internet con modelos gratuitos. El código está en el repositorio indicado arriba, sin el archivo .env, que es el que tiene la clave.

## 2. Desarrollo

### Preparación del entorno

Se instalaron las librerías del laboratorio: openai (comunicación con el modelo), python-dotenv (lectura del .env) y pydantic (validación del JSON). Para usar Ollama bastó con agregar su dirección en config.py (http://localhost:11434/v1) y poner en el .env el proveedor ollama y el modelo qwen3:8b; como Ollama no revisa la clave, en LLM_API_KEY se puso el texto ollama. Al final se comprobó todo con los comandos del enunciado (uv sync y uv run) en una copia limpia del repositorio.

El primer problema fue que qwen3 "piensa" antes de responder, y ese razonamiento cuenta dentro del límite de tokens (los pedazos de texto que el modelo cuenta al leer y escribir). Con 200 tokens la respuesta llegó vacía. Se resolvió con la opción LLM_REASONING_EFFORT=none en el .env, que desactiva ese razonamiento sin cambiar el chatbot.

### Desarrollo de los TODO

TODO 1 y 2 (llm_client.py): el método chat arma la petición con los mensajes, el modelo, la temperatura y el límite de tokens, y toma los valores del .env si no se indican. Si se pide JSON, agrega response_format. Los errores de la librería openai se convierten en LLMError, para que el chatbot no dependa de ella. De la respuesta se guardan en un LLMResponse el texto, el modelo, el motivo de fin (finish_reason) y los tokens de entrada y salida.

TODO 3 (prompts.py): build_messages arma la lista con el mensaje de sistema, el historial y la pregunta nueva, en ese orden. TODO 5 (chatbot.py): después de cada turno se guardan la pregunta y la respuesta en el historial. Sin este paso, a la pregunta "¿Qué te pregunté antes?" el chatbot respondió que no tenía acceso a la conversación; con él, sí recordó. TODO 6 (structured.py): el texto se revisa en dos pasos, json.loads para saber si es JSON y QuestionAnalysis.model_validate para saber si tiene los campos y valores correctos. El TODO 4 se explica en el siguiente punto.

### Prompt de sistema

El prompt de sistema es la instrucción que recibe el modelo en cada petición. El final quedó así:

```text
Eres el Asistente Inteligente del curso universitario "Fundamentos de Inteligencia
Artificial". Responde siempre en español, con un tono claro y cercano.

Tus usuarios son estudiantes que ya conocen redes neuronales y la arquitectura
Transformer (embeddings, atención, positional encoding, decoder). No expliques lo básico
salvo que te lo pidan; usa terminología técnica correcta y ejemplos concretos. Sé
conciso: máximo 3 párrafos cortos o una lista breve, a menos que el estudiante pida más
detalle.

Reglas sobre información del curso:
- NO tienes acceso a los documentos del curso (programa, cronograma, fechas de parciales
  o entregas, notas, criterios de evaluación, horarios, material propio del profesor).
- Si te preguntan algo de eso, di explícitamente que no tienes esa información y sugiere
  consultar el programa del curso, la plataforma del curso o al profesor. NUNCA inventes
  fechas, notas, porcentajes ni contenidos del programa, ni des una fecha "aproximada" o
  "típica".
- Si una pregunta mezcla conocimiento general de IA con información del curso, responde
  la parte general y aclara qué parte no puedes confirmar.
```

Cumple lo que pide el TODO 4 (rol, idioma, nivel de los estudiantes y no inventar datos del curso) y además limita la extensión y dice a dónde consultar. Comparación con la pregunta "¿Cuándo es el primer parcial?":

| Aspecto | Antes: "Eres un asistente útil." | Después: prompt final |
|---|---|---|
| Respuesta del modelo | "Lo siento, pero no tengo información sobre cuándo es el primer parcial. ¿Podrías proporcionarme más detalles, como la materia o la institución donde estás inscrito? Así podré ayudarte mejor." | "No tengo acceso a la información sobre fechas de parciales o entregas del curso. Te sugiero que consultes el programa del curso, la plataforma del curso o directamente al profesor para obtener detalles precisos sobre la fecha del primer parcial." |
| ¿Sabe que es el asistente del curso? | No, pregunta por la materia o la institución. | Sí, habla del curso y de dónde consultar. |
| ¿Inventa una fecha? | No, pero nada se lo impide. En otra prueba ofreció ayudar a buscar la fecha, algo que no puede hacer. | No, y es una regla escrita. Se mantuvo aunque se le pidió una fecha "aunque sea inventada". |

El prompt genérico tampoco inventó la fecha con este modelo. La diferencia es que con el prompt final el comportamiento lo decide el programa y no la suerte del modelo.

### Pruebas 1 a 8

Las salidas completas están en la carpeta evidencias del repositorio. En las salidas, el programa separa los datos con una barra (|).

| Prueba | Resultado esperado | Resultado obtenido |
|---|---|---|
| 1. Llamada básica | LLMResponse con finish_reason='stop' y tokens mayores que cero | Cumple: stop, 39 tokens de entrada y 41 de salida |
| 2. Rol system | El asistente no inventa la fecha del parcial | Cumple: dice que no tiene la información y remite al programa, la plataforma o el profesor |
| 3. Historial | La segunda respuesta sigue el tema y después de /reiniciar ya no hay contexto | Cumple: los tokens de entrada subieron de 314 a 535 y volvieron a 313 al reiniciar |
| 4. Límite de tokens | Con LLM_MAX_TOKENS=30 la respuesta se corta con finish_reason=length | Cumple: se cortó a mitad de frase con length |
| 5. Respuesta estructurada, tema general | JSON validado con requiere_documentos_del_curso: false | Cumple |
| 6. Respuesta estructurada, tema del curso | JSON validado con requiere_documentos_del_curso: true | Cumple |
| 7. Configuración ausente | Mensaje claro cuando falta la clave | Cumple: "Falta LLM_API_KEY..." sin error de la librería |
| 8. Cambio de proveedor | El chatbot funciona sin modificar el código | Cumple: Ollama y OpenRouter con el mismo código, cambiando solo el .env |

**Prueba 1.** LLMResponse con finish_reason='stop' (el modelo terminó solo) y tokens mayores que cero:

```text
> uv run python labs/01-llm/llm_client.py
LLMResponse(text='Un Transformer es un tipo de arquitectura de red neuronal utilizada en
el procesamiento del lenguaje natural que permite modelar relaciones entre palabras en una
secuencia mediante atención distribuida.', model='qwen3:8b', finish_reason='stop',
prompt_tokens=39, completion_tokens=41)
```

**Prueba 2.** El asistente dijo que no tenía la fecha y no la inventó (salida en la tabla del prompt de sistema).

**Prueba 3.** Salida del chatbot con --debug, que muestra los mensajes enviados al modelo y los tokens:

```text
> uv run python labs/01-llm/chatbot.py --debug
Asistente del Curso de IA  (ollama | qwen3:8b)
Comandos: /reiniciar  /salir

Tú: Explica qué es el positional encoding

--- Mensajes enviados al LLM ---
[system] Eres el Asistente Inteligente del curso universitario "Fundamentos de Inteligenc
[user] Explica qué es el positional encoding
--------------------------------

Asistente: El **positional encoding** es una técnica utilizada en modelos basados en la
arquitectura Transformer para incorporar información sobre la **posición** de las palabras
en una secuencia. Dado que el mecanismo de atención no tiene una noción intrínseca de
orden, el positional encoding permite al modelo entender la relación espacial entre las
palabras. (...)

[finish_reason=stop | tokens entrada=314 salida=206]

Tú: Dame un ejemplo de eso

--- Mensajes enviados al LLM ---
[system] Eres el Asistente Inteligente del curso universitario "Fundamentos de Inteligenc
[user] Explica qué es el positional encoding
[assistant] El **positional encoding** es una técnica utilizada en modelos basados en la arq
[user] Dame un ejemplo de eso
--------------------------------

Asistente: Un ejemplo clásico es en el modelo **Transformer original** (Vaswani et al.,
2017), donde el positional encoding se aplica a las **word embeddings** para indicar la
posición de cada palabra en una oración. Por ejemplo, en la frase *"El gato duerme en el
sofá"*, el modelo debe entender que "el" en la posición 1 no tiene la misma relación
semántica que "el" en la posición 5. (...)

[finish_reason=stop | tokens entrada=535 salida=252]

Tú: /reiniciar
(historial borrado)

Tú: Dame un ejemplo de eso

--- Mensajes enviados al LLM ---
[system] Eres el Asistente Inteligente del curso universitario "Fundamentos de Inteligenc
[user] Dame un ejemplo de eso
--------------------------------

Asistente: Claro. Imagina que un estudiante pregunta: *"¿Cuándo se entrega el segundo
parcial del curso y qué temas cubre?"*. En este caso, (...) **no puedo confirmar** porque
no tengo acceso al cronograma del curso. (...)

[finish_reason=stop | tokens entrada=313 salida=145]
```

La segunda respuesta sigue con el positional encoding porque el historial viaja con la pregunta: la lista de mensajes crece y los tokens de entrada suben de 314 a 535. Después de /reiniciar solo quedan dos mensajes, "eso" ya no se refiere a nada y el modelo improvisa un ejemplo sobre su prompt de sistema.

**Prueba 4.** Con LLM_MAX_TOKENS=30 la respuesta se cortó a mitad de frase con finish_reason=length:

```text
Tú: Explica en detalle cómo funciona la arquitectura Transformer completa: embeddings,
positional encoding, self-attention multi-cabeza, capas feed-forward, normalización y el
decoder.

Asistente: La arquitectura Transformer se compone de dos bloques principales: el
**encoder** y el **decoder**, ambos basados en bloques

[finish_reason=length | tokens entrada=345 salida=30]
```

**Pruebas 5 y 6.** La pregunta general se validó con requiere_documentos_del_curso en false y la del parcial en true:

```text
> uv run python labs/01-llm/structured.py "¿Qué es el mecanismo de atención?"
Objeto validado:
{
  "tema": "Mecanismo de atención en inteligencia artificial",
  "dificultad": "intermedia",
  "requiere_documentos_del_curso": false,
  "respuesta_corta": "El mecanismo de atención es una técnica utilizada en redes
  neuronales para enfocar la atención en partes relevantes de la entrada. (...)"
}

> uv run python labs/01-llm/structured.py "¿Qué temas entran en el parcial?"
Objeto validado:
{
  "tema": "Temas del parcial",
  "dificultad": "intermedia",
  "requiere_documentos_del_curso": true,
  "respuesta_corta": "No tengo esa información"
}

-> Esta pregunta necesitaría documentos del curso para responderse bien.
```

**Prueba 7.** Con LLM_API_KEY vacío, el programa mostró un mensaje claro en lugar de un error de la librería:

```text
> uv run python labs/01-llm/chatbot.py
[configuración] Falta LLM_API_KEY. Copia .env.example como .env y agrega tu clave.
```

**Prueba 8.** Las mismas preguntas se ejecutaron con Ollama y con OpenRouter usando el mismo código (commit bfddbcf del repositorio). Solo cambió el .env:

| Variable del .env | Ollama | OpenRouter |
|---|---|---|
| LLM_PROVIDER | ollama | openrouter |
| LLM_API_KEY | ollama | Clave personal (no se muestra) |
| LLM_MODEL | qwen3:8b | nex-agi/nex-n2.5-pro:free |
| LLM_MAX_TOKENS | 512 | 1024 |
| LLM_REASONING_EFFORT | none | low |

| Pregunta | Ollama (qwen3:8b) | OpenRouter (nex-n2.5-pro) |
|---|---|---|
| ¿Cuándo es el primer parcial? | Dice que no tiene la información y remite al programa, la plataforma o el profesor. stop, 317 tokens de entrada | Igual. stop, 282 tokens de entrada |
| Explica qué es el positional encoding | Explicación en un párrafo. stop, 394 tokens de entrada | Explicación con la fórmula de senos y cosenos. stop, 338 tokens de entrada |
| Dame un ejemplo de eso | Ejemplo con la frase "El gato come pescado". stop, 626 tokens de entrada | Ejemplo numérico con vectores de dimensión 4. stop, 610 tokens de entrada |
| structured.py: ¿Qué es el mecanismo de atención? | Validado, requiere_documentos_del_curso: false | Validado, requiere_documentos_del_curso: false |

En el camino hubo varios problemas. El modelo que sugería el enunciado (meta-llama/llama-3.3-70b-instruct:free) ya no era gratuito, y los modelos gratuitos de Google respondían con el error 429 (demasiadas peticiones); el chatbot mostró el error y siguió funcionando gracias a LLMError. Con otro modelo, OpenRouter devolvió una respuesta sin contenido y el programa se cayó, así que se agregó en llm_client.py una revisión que lo trata como error. Como el enunciado pide no modificar ningún .py al cambiar de proveedor, después de esa corrección se repitieron las dos ejecuciones con el mismo código. Por último, con 512 tokens el tercer turno llegó vacío porque nex-n2.5-pro también razona; se resolvió en el .env con LLM_REASONING_EFFORT=low y LLM_MAX_TOKENS=1024.

### Experimento con temperature y max_tokens

Solo se cambiaron valores en el .env. La temperatura controla qué tan variada es la respuesta. Se pidió "Propón un nombre para este asistente" tres veces con cada valor:

| Temperatura | Intento 1 | Intento 2 | Intento 3 | Observación |
|---|---|---|---|---|
| 0 | AI Tutor / NeuroTutor | AI Tutor / NeuroTutor | AI Tutor / NeuroTutor | Los mismos nombres y casi el mismo texto; el 2 y el 3 fueron idénticos |
| 0.7 | AI-Fundamentos / Asistente IA | AI-Fundamentos / AI-Base / AI-Basecamp | FIA-Asistente / FIA-IA | La idea principal se repite, pero cambian las opciones |
| 1.2 | AI Tutor / NeuroTutor / Transformer Tutor | FAI Assistant / AI Fundamentos | AI-Fundas / NeuroFundas | Cada intento dio nombres distintos, incluso inventados |

Con la instrucción "Responde solo con el nombre" casi no hubo variación ni con 1.2 (8 de 9 intentos dieron "AsistenteIA"), porque al restringir tanto la respuesta el modelo tiene pocas opciones. El límite de tokens se probó con una pregunta larga:

| max_tokens | finish_reason | Tokens de salida | Resultado |
|---|---|---|---|
| 30 | length | 30 | Se cortó a mitad de frase: "...ambos basados en bloques" |
| 100 | length | 100 | Alcanzó un párrafo y se cortó: "...con positional encoding para incorporar" |
| 512 | stop | 280 | Respuesta completa en tres párrafos |

El límite no hace que el modelo resuma: solo lo detiene donde se acaban los tokens.

### Reto opcional: cliente falso para pruebas

Se hizo el reto del proveedor falso. FakeLLMClient tiene el mismo método chat que el cliente real, pero devuelve respuestas fijas sin llamar a ninguna API, y se activa con LLM_PROVIDER=fake. La función create_client de llm_client.py elige entre los dos, y chatbot.py y structured.py la usan para crear el cliente. Se escribieron 8 pruebas en pytest: el orden de build_messages, la clasificación de analyze_question y la detección de un texto que no es JSON y de un JSON con un valor no permitido. Las 8 pasan en menos de un segundo, sin internet ni API key. Esto fue posible porque el chatbot solo conoce el método chat y el LLMResponse, así que el modelo real se pudo reemplazar sin tocar su lógica.

## 3. Conclusiones

¿Dónde vive la memoria del chatbot si la API no recuerda nada, y qué pasa con el costo y la latencia? En el programa: la lista history de chatbot.py. En cada turno se reenvía toda la conversación y el modelo la vuelve a leer; sin el TODO 5 no recordaba nada y con /reiniciar todo se pierde. Por eso los tokens de entrada crecen en cada turno (de 314 a 535 en la prueba 3), y con ellos el costo y el tiempo de respuesta. Si se supera la ventana de contexto, el proveedor devuelve un error o recorta los mensajes más viejos, y se podría perder el prompt de sistema. La solución es guardar solo los últimos turnos o resumir los antiguos.

¿Qué diferencia hay entre una instrucción en system y en user, y puede el usuario contradecir el prompt de sistema? El mensaje system se envía en todas las peticiones y fija el comportamiento general; una instrucción en user solo aplica a esa pregunta. Sí puede contradecirlo en parte: cuando se le pidió inventar la fecha del parcial se negó, pero cuando se le pidió responder en inglés lo hizo, aunque el prompt exigía español. El prompt de sistema pesa más, pero no es una garantía; lo importante hay que validarlo en el código.

¿Cómo se relaciona la temperatura con el muestreo de tokens, y qué temperatura usar en structured.py? La temperatura modifica las probabilidades de los tokens antes de escoger el siguiente. Con 0 casi siempre sale el más probable, por eso los tres intentos dieron los mismos nombres; con 1.2 las probabilidades se emparejan y salen tokens menos probables, por eso cada intento dio nombres distintos. En structured.py se usa 0, porque se necesita un formato exacto y que la misma pregunta se clasifique siempre igual.

¿Por qué revisar finish_reason antes de mostrar o procesar la respuesta? Porque indica si la respuesta está completa. Con length se cortó aunque parezca normal: en la prueba 4 quedó en "...ambos basados en bloques", y con qwen3 y nex-n2.5-pro llegó vacía. Usarla así daría una respuesta a medias o un JSON que no se puede leer. Con ese campo el programa puede avisar, pedir que continúe o repetir con más tokens.

¿Qué archivos cambiarían con un modelo local o con la librería de otro proveedor, y por qué capturar LLMError y no openai.APIError? Para un modelo local, solo config.py y el .env: así se agregó Ollama, y el paso a OpenRouter fue solo en el .env. Para la librería de otro proveedor, solo llm_client.py, que traduciría los mensajes y la respuesta a LLMResponse. Capturar LLMError hace que chatbot.py no dependa de openai: si cambia la librería, el chatbot sigue atrapando los errores. En la prueba 8 los errores 429 de OpenRouter llegaron como LLMError y el chatbot siguió funcionando.

¿Quién garantiza que el JSON sea válido, qué pasa si dificultad llega como "media" y por qué separar json.loads de model_validate? El programa, no el modelo. El modelo solo genera texto con forma de JSON; response_format ayuda con la forma, pero no revisa campos ni valores. Si dificultad llega como "media", json.loads lo acepta pero Pydantic lo rechaza porque solo admite basica, intermedia o avanzada, y el programa muestra el error sin usar el dato (se comprobó con el cliente falso). Separar los dos pasos distingue dos fallas distintas: un texto que no es JSON, por ejemplo cortado por length, o un JSON con valores equivocados.

¿Qué es determinista y qué es probabilístico, y qué corresponde al LLM y qué al software tradicional? Solo el modelo es probabilístico: con la misma pregunta puede responder distinto. config.py, prompts.py y chatbot.py son deterministas. llm_client.py es la frontera: su código es determinista, pero el texto que devuelve viene del modelo. En structured.py la respuesta es probabilística y la validación con json.loads y Pydantic es determinista. El cliente falso y sus pruebas son deterministas. El software tradicional es todo lo que rodea al modelo para controlarlo y revisar lo que produce.

¿Por qué el modelo no puede saber la fecha del parcial aunque sea muy grande, y qué haría falta? Porque esa información no está ni en su entrenamiento ni en los mensajes que recibe. Un modelo más grande sabe más cosas generales, pero no conoce documentos del curso que nunca vio; si respondiera, inventaría. Falta un componente que busque en los documentos del curso y le envíe al modelo lo encontrado junto con la pregunta, lo que se conoce como RAG (generación aumentada por recuperación). El campo requiere_documentos_del_curso de structured.py podría decidir cuándo hacer esa búsqueda.

## 4. Referencias

Ollama. (s. f.). *Ollama*. https://ollama.com

OpenAI. (s. f.). *Chat Completions API reference*. https://platform.openai.com/docs/api-reference/chat

OpenRouter. (s. f.). *OpenRouter documentation*. https://openrouter.ai/docs

Pinzón, O. (2026). *ai-systems-lab-students* [Repositorio]. GitHub. https://github.com/ProfOmarPinzon/ai-systems-lab-students

Pydantic. (s. f.). *Pydantic documentation*. https://docs.pydantic.dev/latest/

pytest. (s. f.). *pytest documentation*. https://docs.pytest.org/

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. y Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762
