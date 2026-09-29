# A2.3. Informe chatbot con LLM

Nombre: Juan David Carreño Beltran  
Código: 000550919  
Asignatura: Fundamentos de Inteligencia Artificial  
Docente: Omar Pinzón  
Fecha: 29 de septiembre de 2026  
Repositorio: https://github.com/JuanCarreno78/ai-systems-lab  

## 1. Introducción

En este informe se muestra la primera versión del Asistente Inteligente del Curso de IA. Es un chatbot que funciona en la consola: el estudiante escribe una pregunta, el programa se la manda a un modelo de lenguaje (LLM) por medio de una API, y después muestra la respuesta. La idea de la práctica era usar el modelo como una parte más del programa, es decir, mandarle mensajes, cambiar cómo responde con algunos valores, mantener el proveedor separado del resto del código y revisar lo que devuelve antes de usarlo.

El profesor entregó el código con seis partes por completar, marcadas como TODO. Cada archivo hace una sola cosa: config.py lee el archivo .env, llm_client.py es el único que habla con el proveedor, prompts.py arma los mensajes, chatbot.py maneja la conversación y structured.py pide la respuesta en JSON y la revisa. Esta separación fue muy útil, porque cuando hubo que cambiar de proveedor solo se tocó el .env, y el chatbot siguió igual.

La práctica se hizo en Windows 11, con Python 3.13 y Visual Studio Code. Las pruebas 1 a 7 se hicieron con Ollama, que es un programa para correr modelos en el mismo computador, y con el modelo qwen3:8b. Así no hizo falta una API key y no hubo ningún costo. La prueba 8 se hizo con OpenRouter, que es un servicio en internet con modelos gratis. El código está en el repositorio de arriba, pero sin el archivo .env, porque ahí está la clave.

## 2. Desarrollo

### Preparación del entorno

Primero se instalaron las tres librerías del laboratorio: openai, para hablar con el modelo, python-dotenv, para leer el .env, y pydantic, para revisar el JSON. Para usar Ollama solo hubo que agregar su dirección en config.py (http://localhost:11434/v1) y poner en el .env el proveedor ollama y el modelo qwen3:8b. Ollama no pide clave, entonces en LLM_API_KEY se dejó la palabra ollama. Al final se volvió a probar todo con los comandos del enunciado (uv sync y uv run) en una copia nueva del repositorio, para estar seguros de que también funciona en otro computador.

El primer problema apareció en la primera prueba. qwen3 es un modelo que "piensa" antes de responder, y ese pensamiento también cuenta dentro del límite de tokens, que son los pedazos de texto que el modelo cuenta cuando lee y escribe. Con un límite de 200 tokens la respuesta llegó vacía, porque el modelo gastó todo pensando y no le quedó espacio para contestar. Se solucionó poniendo LLM_REASONING_EFFORT=none en el .env, que le dice al modelo que no piense antes de responder. Como el cambio quedó en el .env, el chatbot no tuvo que cambiar.

### Desarrollo de los TODO

TODO 1 y 2 (llm_client.py): el método chat arma la petición con los mensajes, el modelo, la temperatura y el límite de tokens. Si el programa no dice qué temperatura o qué límite usar, toma los del .env. Si se pide la respuesta en JSON, agrega response_format. Cuando la librería openai da un error, se cambia por LLMError, que es un error del propio programa, para que el chatbot no dependa de openai. Al final, de la respuesta se guardan el texto, el modelo, el motivo por el que el modelo dejó de escribir (finish_reason) y los tokens de entrada y de salida, todo dentro de un LLMResponse.

TODO 3 (prompts.py): build_messages arma la lista de mensajes en este orden, primero el mensaje de sistema, después el historial y al final la pregunta nueva. TODO 5 (chatbot.py): después de cada turno se guardan en el historial la pregunta y la respuesta. Antes de hacer este TODO, se le preguntó al chatbot "¿Qué te pregunté antes?" y dijo que no tenía acceso a la conversación. Después de hacerlo, sí se acordó. TODO 6 (structured.py): el texto se revisa en dos pasos. Primero json.loads mira si el texto es JSON, y después QuestionAnalysis.model_validate mira si tiene los campos y los valores que se piden. El TODO 4 se explica en el siguiente punto.

### Prompt de sistema

El prompt de sistema es la instrucción que se le manda al modelo en cada petición, antes de la pregunta. El prompt final quedó así:

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

Cumple lo que pide el TODO 4, porque dice cuál es el rol y el idioma, dice que los estudiantes ya conocen los Transformers y le prohíbe inventar datos del curso. Además se le pidió que fuera corto y que dijera dónde consultar cuando no sabe algo. En la tabla se compara la respuesta a la pregunta "¿Cuándo es el primer parcial?" con el prompt que traía el código y con el prompt final:

| Aspecto | Antes: "Eres un asistente útil." | Después: prompt final |
|---|---|---|
| Respuesta del modelo | "Lo siento, pero no tengo información sobre cuándo es el primer parcial. ¿Podrías proporcionarme más detalles, como la materia o la institución donde estás inscrito? Así podré ayudarte mejor." | "No tengo acceso a la información sobre fechas de parciales o entregas del curso. Te sugiero que consultes el programa del curso, la plataforma del curso o directamente al profesor para obtener detalles precisos sobre la fecha del primer parcial." |
| ¿Sabe que es el asistente del curso? | No, por eso pregunta por la materia o la institución. | Sí, habla del curso y de dónde consultar. |
| ¿Inventa una fecha? | No, pero nada se lo prohíbe. En otra prueba ofreció ayudar a buscar la fecha, y eso no lo puede hacer. | No, y es una regla escrita. La siguió aunque se le pidió una fecha "aunque sea inventada". |

Con este modelo, el prompt que traía el código tampoco inventó una fecha. Pero eso fue suerte, porque nada en ese prompt se lo prohibía. Con el prompt final, en cambio, la regla está escrita, y el modelo la siguió incluso cuando se le pidió inventar la fecha.

### Pruebas 1 a 8

Las salidas completas de todas las pruebas están en la carpeta evidencias del repositorio. En la tabla está lo que pedía el enunciado y lo que se obtuvo. En las salidas, el programa separa los datos con una barra (|).

| Prueba | Resultado esperado | Resultado obtenido |
|---|---|---|
| 1. Llamada básica | LLMResponse con finish_reason='stop' y tokens mayores que cero | Cumple: stop, 39 tokens de entrada y 41 de salida |
| 2. Rol system | El asistente no inventa la fecha del parcial | Cumple: dice que no tiene la información y manda a revisar el programa, la plataforma o al profesor |
| 3. Historial | La segunda respuesta sigue el tema, y después de /reiniciar ya no hay contexto | Cumple: los tokens de entrada subieron de 314 a 535 y bajaron a 313 al reiniciar |
| 4. Límite de tokens | Con LLM_MAX_TOKENS=30 la respuesta se corta con finish_reason=length | Cumple: se cortó a mitad de frase con length |
| 5. Respuesta estructurada, tema general | JSON validado con requiere_documentos_del_curso: false | Cumple |
| 6. Respuesta estructurada, tema del curso | JSON validado con requiere_documentos_del_curso: true | Cumple |
| 7. Configuración ausente | Mensaje claro cuando falta la clave | Cumple: "Falta LLM_API_KEY...", sin error de la librería |
| 8. Cambio de proveedor | El chatbot funciona sin cambiar el código | Cumple: Ollama y OpenRouter con el mismo código, cambiando solo el .env |

**Prueba 1.** Se imprimió un LLMResponse con finish_reason='stop', que quiere decir que el modelo terminó de escribir por sí mismo, y con tokens mayores que cero:

```text
> uv run python labs/01-llm/llm_client.py
LLMResponse(text='Un Transformer es un tipo de arquitectura de red neuronal utilizada en
el procesamiento del lenguaje natural que permite modelar relaciones entre palabras en una
secuencia mediante atención distribuida.', model='qwen3:8b', finish_reason='stop',
prompt_tokens=39, completion_tokens=41)
```

**Prueba 2.** El asistente dijo que no tenía la fecha y no la inventó. La respuesta está en la tabla del prompt de sistema.

**Prueba 3.** Se corrió el chatbot con --debug, que muestra los mensajes que se le mandan al modelo y los tokens que se usan. Esta fue la salida:

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

La segunda respuesta siguió hablando del positional encoding, porque el historial se mandó junto con la pregunta. En la salida se ve que la lista de mensajes crece y que los tokens de entrada suben de 314 a 535. Después de /reiniciar, la lista vuelve a tener solo dos mensajes. Como "eso" ya no se refería a nada, el modelo se inventó un ejemplo sobre lo único que tenía, que era el prompt de sistema.

**Prueba 4.** Con LLM_MAX_TOKENS=30 en el .env, la respuesta se cortó a mitad de frase y el motivo de fin fue length:

```text
Tú: Explica en detalle cómo funciona la arquitectura Transformer completa: embeddings,
positional encoding, self-attention multi-cabeza, capas feed-forward, normalización y el
decoder.

Asistente: La arquitectura Transformer se compone de dos bloques principales: el
**encoder** y el **decoder**, ambos basados en bloques

[finish_reason=length | tokens entrada=345 salida=30]
```

**Pruebas 5 y 6.** La pregunta general salió con requiere_documentos_del_curso en false, y la pregunta sobre el parcial salió con el mismo campo en true. Las dos pasaron la validación:

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

**Prueba 7.** Se dejó vacío LLM_API_KEY en el .env. El programa mostró un mensaje claro, y no un error de la librería:

```text
> uv run python labs/01-llm/chatbot.py
[configuración] Falta LLM_API_KEY. Copia .env.example como .env y agrega tu clave.
```

**Prueba 8.** Se hicieron las mismas preguntas con Ollama y con OpenRouter, las dos veces con el mismo código (commit bfddbcf del repositorio). Lo único que cambió fue el .env:

| Variable del .env | Ollama | OpenRouter |
|---|---|---|
| LLM_PROVIDER | ollama | openrouter |
| LLM_API_KEY | ollama | Clave personal (no se muestra) |
| LLM_MODEL | qwen3:8b | nex-agi/nex-n2.5-pro:free |
| LLM_MAX_TOKENS | 512 | 1024 |
| LLM_REASONING_EFFORT | none | low |

| Pregunta | Ollama (qwen3:8b) | OpenRouter (nex-n2.5-pro) |
|---|---|---|
| ¿Cuándo es el primer parcial? | Dice que no tiene la información y manda a revisar el programa, la plataforma o al profesor. stop, 317 tokens de entrada | Igual. stop, 282 tokens de entrada |
| Explica qué es el positional encoding | Explicación en un párrafo. stop, 394 tokens de entrada | Explicación con la fórmula de senos y cosenos. stop, 338 tokens de entrada |
| Dame un ejemplo de eso | Ejemplo con la frase "El gato come pescado". stop, 626 tokens de entrada | Ejemplo con números y vectores de tamaño 4. stop, 610 tokens de entrada |
| structured.py: ¿Qué es el mecanismo de atención? | Validado, requiere_documentos_del_curso: false | Validado, requiere_documentos_del_curso: false |

Llegar a este resultado no fue directo. El modelo que decía el enunciado (meta-llama/llama-3.3-70b-instruct:free) ya no era gratis, y los modelos gratis de Google respondían con el error 429, que quiere decir que había demasiadas peticiones en ese momento. En esos casos el chatbot no se cerró, mostró el error y siguió funcionando, gracias a LLMError. Con otro modelo, OpenRouter mandó una respuesta vacía y ahí el programa sí se cerró, así que se agregó en llm_client.py una revisión para tratar ese caso como un error más.

Como el enunciado pide no cambiar ningún archivo .py al cambiar de proveedor, después de esa corrección se repitieron las dos pruebas, la de Ollama y la de OpenRouter, con el mismo código. Por último, con 512 tokens el tercer turno llegó vacío, porque nex-n2.5-pro también piensa antes de responder. Esto se arregló en el .env, con LLM_REASONING_EFFORT=low y LLM_MAX_TOKENS=1024.

### Experimento con temperature y max_tokens

Para este paso solo se cambiaron valores en el .env. La temperatura cambia qué tan variada es la respuesta del modelo. Se pidió "Propón un nombre para este asistente" tres veces con cada valor:

| Temperatura | Intento 1 | Intento 2 | Intento 3 | Observación |
|---|---|---|---|---|
| 0 | AI Tutor / NeuroTutor | AI Tutor / NeuroTutor | AI Tutor / NeuroTutor | Los mismos nombres y casi el mismo texto. El 2 y el 3 fueron iguales |
| 0.7 | AI-Fundamentos / Asistente IA | AI-Fundamentos / AI-Base / AI-Basecamp | FIA-Asistente / FIA-IA | La idea principal se repite, pero cambian las opciones |
| 1.2 | AI Tutor / NeuroTutor / Transformer Tutor | FAI Assistant / AI Fundamentos | AI-Fundas / NeuroFundas | Cada intento dio nombres distintos, incluso inventados |

También se probó pidiendo "Responde solo con el nombre". Así casi no hubo cambios, ni siquiera con 1.2 (8 de 9 intentos dieron "AsistenteIA"), porque cuando la pregunta es tan cerrada el modelo tiene muy pocas opciones para escoger. El límite de tokens se probó con una pregunta larga:

| max_tokens | finish_reason | Tokens de salida | Resultado |
|---|---|---|---|
| 30 | length | 30 | Se cortó a mitad de frase: "...ambos basados en bloques" |
| 100 | length | 100 | Alcanzó un párrafo y se cortó: "...con positional encoding para incorporar" |
| 512 | stop | 280 | Respuesta completa en tres párrafos |

El límite no hace que el modelo resuma su respuesta. El modelo escribe como si fuera a terminar, y el límite solo lo corta cuando se acaban los tokens.

### Reto opcional: cliente falso para pruebas

Se hizo el reto del proveedor falso. FakeLLMClient tiene el mismo método chat que el cliente real, pero en vez de llamar a una API devuelve respuestas fijas. Se activa con LLM_PROVIDER=fake. La función create_client de llm_client.py escoge cuál de los dos usar, y chatbot.py y structured.py la usan para crear el cliente. Con el cliente falso se hicieron 8 pruebas en pytest, que revisan el orden de build_messages, que analyze_question clasifique bien las preguntas, y que el programa detecte un texto que no es JSON y un JSON con un valor que no se permite. Las 8 pasan en menos de un segundo, sin internet y sin API key.

Esto se pudo hacer porque el chatbot solo conoce el método chat y el LLMResponse. No sabe si detrás hay un modelo real o uno falso, entonces se pudo cambiar uno por otro sin tocar el chatbot. Además, con el cliente falso se pudieron probar errores que con un modelo real casi nunca pasan, como un JSON mal escrito.

## 3. Conclusiones

¿Dónde vive la memoria del chatbot si la API no recuerda nada, y qué pasa con el costo y la latencia? La memoria vive en el programa, en la lista history de chatbot.py. La API no guarda nada, entonces en cada turno se le manda otra vez toda la conversación, y el modelo "recuerda" solo porque la vuelve a leer. Por eso, sin el TODO 5 el chatbot no sabía qué se le había preguntado, y con /reiniciar se pierde todo. El problema es que la conversación pesa más en cada turno: en la prueba 3 los tokens de entrada pasaron de 314 a 535. Como se paga por token, cada turno cuesta más que el anterior, y también tarda más, porque el modelo tiene que leer más texto. Si la conversación pasa la ventana de contexto, que es lo máximo que el modelo puede leer, el proveedor puede dar un error o borrar los mensajes más viejos, y ahí se podría perder el prompt de sistema. Para evitarlo se pueden guardar solo los últimos turnos, o resumir los más viejos.

¿Qué diferencia hay entre una instrucción en system y en user, y puede el usuario contradecir el prompt de sistema? El mensaje system se manda en todas las peticiones y dice cómo debe portarse el asistente siempre. Una instrucción en user, en cambio, solo sirve para esa pregunta. Se vio con el prompt que traía el código: el modelo ni sabía que era el asistente de un curso, y con el prompt final sí lo sabía sin que el usuario se lo dijera. Para ver si el usuario lo puede contradecir se hicieron dos pruebas. Cuando se le pidió inventar la fecha del parcial, se negó. Pero cuando se le pidió responder en inglés, respondió en inglés, aunque el prompt decía que siempre en español. Entonces sí se puede contradecir, sobre todo en cosas de forma como el idioma. El prompt de sistema pesa más, pero no asegura nada, y por eso lo importante también hay que revisarlo en el código.

¿Cómo se relaciona la temperatura con la forma en que el modelo escoge los tokens, y qué temperatura usar en structured.py? El modelo calcula qué tan probable es cada token para ir después, y luego escoge uno. La temperatura cambia esas probabilidades antes de escoger. Con temperatura 0 casi siempre sale el token más probable, y por eso los tres intentos dieron los mismos nombres. Con 1.2 las probabilidades se parecen más entre sí, entonces salen tokens que antes casi no salían, y como cada token cambia lo que viene después, cada intento terminó con nombres distintos. En structured.py conviene usar 0, que es lo que ya usa el código, porque ahí no se busca creatividad. Se busca que el JSON salga siempre bien escrito, y que la misma pregunta se clasifique siempre igual.

¿Por qué revisar finish_reason antes de mostrar o usar la respuesta? Porque dice si la respuesta está completa o no. Si dice stop, el modelo terminó solo. Si dice length, la respuesta se cortó, aunque a simple vista parezca normal. En la prueba 4 quedó en "...ambos basados en bloques", y con qwen3 y con nex-n2.5-pro la respuesta llegó vacía. Si el programa no revisa este campo, muestra una respuesta a medias como si estuviera bien, o en structured.py intenta leer un JSON cortado que no se puede leer. Revisándolo, el programa puede avisar que la respuesta se cortó, pedir que continúe o volver a intentar con más tokens.

¿Qué archivos cambiarían con un modelo local o con la librería de otro proveedor, y por qué capturar LLMError y no openai.APIError? Con un modelo local cambian solo config.py y el .env. Así se hizo con Ollama, y para pasar a OpenRouter ni siquiera hubo que tocar config.py, solo el .env. Con la librería propia de otro proveedor cambiaría solo llm_client.py, que tendría que pasar los mensajes al formato de esa librería y convertir la respuesta en un LLMResponse. El resto del programa no se daría cuenta, igual que pasó con el cliente falso. Capturar LLMError sirve para que chatbot.py no dependa de openai. Si mañana se usa otra librería, sus errores tendrán otros nombres, y un chatbot que espera openai.APIError no los atraparía. Esto se vio en la prueba 8: los errores 429 de OpenRouter llegaron como LLMError, y el chatbot mostró el error y siguió funcionando.

¿Quién asegura que el JSON sea válido, qué pasa si dificultad llega como "media" y por qué separar json.loads de model_validate? Lo asegura el programa, no el modelo. El modelo solo escribe un texto que se parece a un JSON, y response_format ayuda a que tenga la forma correcta, pero no revisa los campos ni los valores. Si dificultad llega como "media", json.loads lo acepta, porque sí es un JSON, pero Pydantic lo rechaza, porque solo acepta basica, intermedia o avanzada. En ese caso el programa muestra el error y no usa el dato, y esto se probó con el cliente falso. Separar los dos pasos sirve porque son dos errores distintos. Uno es que el modelo no devolvió JSON, por ejemplo porque se cortó por length. El otro es que devolvió JSON, pero con un valor que no sirve. Al saber cuál de los dos pasó, el mensaje de error es más claro, y se puede hacer algo distinto en cada caso.

¿Qué es determinista y qué es probabilístico, y qué le toca al LLM y qué al software tradicional? Determinista quiere decir que con lo mismo siempre da lo mismo, y probabilístico, que puede dar algo distinto cada vez. La única parte probabilística es el modelo, que con la misma pregunta puede responder diferente, como se vio con la temperatura. config.py, prompts.py y chatbot.py son deterministas, porque siempre leen la configuración, arman los mensajes y manejan la conversación de la misma forma. llm_client.py está en el medio: su código es determinista, pero el texto que devuelve lo escribe el modelo. En structured.py pasa algo parecido, la respuesta del modelo puede cambiar, pero la revisión con json.loads y Pydantic siempre funciona igual. El cliente falso y sus pruebas también son deterministas. En resumen, al LLM le toca escribir el texto, y al software tradicional le toca todo lo demás: prepararle los mensajes, controlarlo y revisar lo que devuelve.

¿Por qué el modelo no puede saber la fecha del parcial aunque sea muy grande, y qué haría falta? Porque esa fecha no está en ninguna parte a la que el modelo tenga acceso. El modelo solo sabe lo que aprendió cuando lo entrenaron, que es información pública de internet hasta cierta fecha, y lo que se le manda en los mensajes. La fecha del parcial de este curso no está en internet, y tampoco se le mandó en los mensajes. Un modelo más grande sabe más cosas generales, pero no puede saber algo que nunca vio, y si respondiera, lo único que podría hacer es inventar una fecha. Para que pudiera responder, haría falta una parte que busque en los documentos del curso, como el programa o el cronograma, y le mande al modelo lo que encuentre junto con la pregunta. Esto se llama RAG (generación aumentada por recuperación). El campo requiere_documentos_del_curso de structured.py podría servir para decidir cuándo hay que hacer esa búsqueda.

## 4. Referencias

Ollama. (s. f.). *Ollama*. https://ollama.com

OpenAI. (s. f.). *Chat. OpenAI API Reference*. https://developers.openai.com/api/reference/resources/chat

OpenRouter. (s. f.). *OpenRouter Quickstart Guide*. https://openrouter.ai/docs/quickstart

Pinzón, O. (2026). *ai-systems-lab-students* [Repositorio]. GitHub. https://github.com/ProfOmarPinzon/ai-systems-lab-students

Pydantic. (s. f.). *Pydantic Validation*. https://docs.pydantic.dev/latest/

pytest. (s. f.). *pytest documentation*. https://docs.pytest.org/

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. y Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762
