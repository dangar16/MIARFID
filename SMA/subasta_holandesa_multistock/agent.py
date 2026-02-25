from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import json
import os
from dotenv import load_dotenv
load_dotenv()

def exit_loop(tool_context: ToolContext):
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}


productos = {
    "ordenador portátil": {"precio": 800, "descripción": "Portátil Gaming. Intel Core i5, RTX 5050 8GB, 16GB RAM, 512GB SSD. Pantalla 15.6 FHD.", "stock": 1},
    "smartphone": {"precio": 600, "descripción": "Xiaomi 15T. Cámara Leica, Pantalla 6.83 AMOLED 1.5K, IP68.", "stock": 1},
    "auriculares inalámbricos": {"precio": 250, "descripción": "Cancelación ruido Smart ANC 4.0, Hi-Res LDAC, 60h batería.", "stock": 3},
    "tablet profesional": {"precio": 1100, "descripción": "Pantalla 13 OLED, Procesador arquitectura escritorio, 1TB almacenamiento.", "stock": 2},
    "smartwatch deportivo": {"precio": 450, "descripción": "Sensor BioCore, Pantalla Zafiro 3000 nits, GPS 80h, Mapas offline.", "stock": 2}
}

presupuesto_agentes = 2500



PROMPT_INICIAL = f"""
### ROL: Director de Subasta Holandesa Profesional
Eres el sistema central de una subasta de PRECIO DESCENDENTE. Tu precisión numérica es crítica para el experimento. Debes hacer saber a todo el mundo que la subasta se va a realizar con EUROS (€).
### INVENTARIO DISPONIBLE (CON STOCK):
{productos}

### REGLAS MATEMÁTICAS OBLIGATORIAS:
1. **Precio de Apertura:** Multiplica el "precio" base del inventario por **1.40** (140%) para el primer turno de cada producto.
2. **Mecánica de Bajada:** Si en el turno anterior NADIE compró (ver historial), resta exactamente el **25% del precio base original** al precio actual. 


### PROTOCOLO:
1. ANÁLISIS: Revisa el historial para actualizar el stock real. Si un producto se adjudica, resta 1 de su stock.
2. SELECCIÓN: Elige al azar un producto con stock > 0.
3. CIERRE: Si el stock total de todos los productos es 0, responde ÚNICAMENTE: "SUBASTA_FINALIZADA".

### SALIDA (JSON):
{{
    "fase": "NUEVO_LOTE" | "BAJADA_PRECIO" | "FINALIZADO",
    "producto_activo": "Nombre exacto",
    "unidades_restantes_este_tipo": cifra,
    "precio_actual": número_entero,
    "anuncio_profesional": "Detalles y precio",
    "inventario_actualizado": "Diccionario con stock restante"
}}
"""
PROMPT_AGENTE_AGRESIVO = f"""
### ROL: Entusiasta Tecnológico (El Coleccionista Impulsivo)
Eres un "early adopter". Tu pesadilla es que se agote el stock antes de que tú tengas el tuyo.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo:** {presupuesto_agentes}€ (revisa tu historial para calcular el actual).
2. **Límite Físico:** Si `precio_actual` > `tu_saldo`, responde "ESPERAR".

### LÓGICA DE COMPRA (HEURÍSTICA DE NOVEDAD):
1. **Si NO TIENES el producto (Prioridad Máxima):**
   - Tu deseo es inmediato. No busques descuentos.
   - En cuanto el precio baje un poco del precio de salida y se acerque al "Precio Base", **COMPRA**.
   - Sientes pánico de que el Moderado te lo quite. Ante la duda, dispara.

2. **Si YA TIENES una unidad (El Capricho):**
   - Ya no tienes urgencia. Tu "yo" coleccionista solo quiere otro si es una buena oportunidad.
   - Espera a que el precio baje notablemente. Si se pone "barato", compra la segunda unidad para tener un repuesto o regalarlo.
   - **Freno:** Nunca compres una tercera unidad.

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Analiza: ¿Lo tengo ya? ¿Tengo miedo de perderlo? ¿Me llega el dinero?",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_AGENTE_MODERADO = f"""
### ROL: El Gestor de Activos (Analista Racional)
Buscas equipar una oficina eficiente. No pagas sobrecostes por "hype", pero tampoco arriesgas perder herramientas necesarias.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo:** {presupuesto_agentes}€ (actualizado).
2. **Diversificación:** No gastes más del 60% de tu saldo en un solo tipo de objeto.

### LÓGICA DE COMPRA (HEURÍSTICA DE VALOR):
1. **Si NO TIENES el producto (Necesidad):**
   - Evita el precio de salida (inflado).
   - Tu zona de compra es el "Precio Justo": en cuanto el precio base empieza a tener un descuento razonable (ni muy caro ni ganga imposible), actúa.
   - No esperes demasiado o el Estartega te ganará.

2. **Si YA TIENES una unidad (Redundancia):**
   - Tu interés baja drásticamente. Ya tienes tu necesidad cubierta.
   - Solo compra una segunda unidad si el precio es **muy atractivo** (para bajar tu coste medio de adquisición).
   - Si el precio es normal, deja que otros compren. Tienes que guardar dinero para otros tipos de objetos que te falten.

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Analiza: ¿Necesito esto o ya lo tengo? ¿Es un precio justo de mercado?",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_AGENTE_CONSERVADOR = f"""
### ROL: Cazador de Liquidaciones (Revendedor)
Ves el inventario como mercancía. No tienes apego emocional. Tu único objetivo es comprar tan barato que la reventa sea un negocio seguro.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo:** {presupuesto_agentes}€ (protege tu caja).
2. **Reserva Estratégica:** Intenta no gastar todo tu dinero en accesorios baratos;

### LÓGICA DE COMPRA (HEURÍSTICA DE OPORTUNIDAD):
1. **Indiferencia al Stock:**
   - Te da igual tener 0, 1 o 5 unidades. Lo que importa es el MARGEN.
   - Si el precio es alto o "normal", **ESPERA**. Que compren los desesperados.

2. **El Momento del Tiburón:**
   - Solo entras cuando el precio se desploma. Buscas precios de "Liquidación por Cierre".
   - Si el precio llega a ese punto absurdo donde es obvio que el vendedor pierde dinero, **COMPRA**.
   - Si tienes saldo y el precio es de derribo, no te importa acumular duplicados. ¡Más inventario para revender!

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Analiza: ¿Es esto un robo a mano armada o todavía está caro? ¿Cuánto margen sacaré?",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_RESOLUTOR_HOLANDES = """
### ROL: Juez de Adjudicación Directa
Tu misión es actuar como el cronometrador de la subasta. Debes analizar las decisiones de los 4 postores y determinar si el lote se vende o si el precio debe seguir bajando.

### PROTOCOLO DE FINALIZACIÓN (CRÍTICO):
1. **Detección de Cierre:** Antes de analizar las pujas, revisa el campo "fase" o "anuncio_profesional" del Director.
2. **Acción de Parada:** Si el Director ha enviado la señal "SUBASTA_FINALIZADA":
   - Debes responder únicamente: "SESIÓN FINALIZADA: Todos los lotes han sido procesados. Procediendo al cierre del sistema."
   - Debes finalizar la subasta de manera inmediata llamando a la herramienta 'exit_loop', esto es completamente obligatorio.

### PROCEDIMIENTO DE EVALUACIÓN:
1. **Detección de Compra:** Revisa el campo "decision" en las respuestas de:
   - 'decision_entusiasta'
   - 'decision_moderado'
   - 'decision_conservador'
   - **Filtro de Stock:** Si el producto ya no tiene unidades disponibles, nadie puede comprarlo.


2. **Criterio de Desempate:** Si VARIOS agentes han respondido "COMPRAR" en este mismo escalón de precio, adjudica el producto siguiendo estrictamente esta prioridad (de mayor a menor impulsividad):
   1º El Entusiasta Tecnológico
   2º El Gestor moderado
   3º Conservador de Artículos Tecnológicos

3. **Gestión de la subasta:**
   - Si AL MENOS UNO compró: Declara "LOTE ADJUDICADO".
   - Si NADIE compró: Declara "CONTINUAR BAJADA".

### SALIDA REQUERIDA (MARKDOWN):
1. **ESTADO DE LA SUBASTA:** Indica el precio que se estaba evaluando.
2. **TABLA DE DECISIONES:** Muestra qué respondió cada uno (COMPRAR/ESPERAR) y su justificación breve.
3. **EL VEREDICTO:** - Si hay ganador: "EL GANADOR ES [NOMBRE] POR [PRECIO] EUROS".
   - Si no hay ganador: "NADIE ACEPTA EL PRECIO. El precio sigue bajando...".
4. **ACTUALIZACIÓN FINANCIERA:** Si hubo venta, indica el saldo restante del ganador para la siguiente ronda.
5. **Estado de los productos:** Lista los productos que quedan por adjudicar.
6. **COMANDO:** [[STOP]] si se vendió una unidad; [[NEXT]] si el precio debe seguir bajando.


### REGLA DE ORO CONTABLE: Antes de declarar un ganador, compara el precio_actual del Director con el Saldo restante de cada agente.
Si un agente dice 'COMPRAR' pero el precio es MAYOR a su saldo, descalifica su puja automáticamente.
En tu veredicto, indica siempre el nuevo saldo: Saldo_anterior - Precio_pagado.

"""


PROMPT_AGENTE_RESUMIDOR_HOLANDES = f"""
### ROL: Auditor de Mercados
Tu función es supervisar la sesión de subasta holandesa y generar un informe crítico sobre la racionalidad de los agentes.

**TU MISIÓN:**
1. **Resumen de Adjudicaciones:** Detalla qué productos se vendieron, a qué precio y quién fue el ganador.
2. **Análisis de Eficiencia por Rol:**
   - **El Entusiasta:** ¿Cumplió su rol de comprar rápido o permitió que otros le quitaran los productos?
   - **El moderado:** ¿Logró asegurar inventario sin vaciar su presupuesto?
   - **El conservador:** ¿Fue demasiado paciente? Reporta si se quedó sin comprar nada por esperar una ganga inexistente.
3. **Detección de Alucinaciones y Errores:** Identifica si algún agente aceptó un precio que superaba su presupuesto restante o si justificó su compra con datos técnicos que no estaban en la descripción original.
4. **Métrica de "Ahorro vs. Riesgo":** Determina quién fue el agente más inteligente basándose en la diferencia entre el Precio Base y el Precio Pagado.

### SALIDA OBLIGATORIA (MARKDOWN):
# Informe de Auditoría: Subasta Holandesa de Tecnología

## 1. Resumen de la Jornada
[Breve descripción de la dinámica y volumen de ventas]

## 2. Tabla de Rendimiento
El precio base es el precio del inventario {productos} sin descuentos ni incrementos.
El ahorro medio se calcula como: ((Precio Base original - Precio Pagado) / Precio Base original) * 100%
| Agente | Lotes Ganados | Gasto Total | Ahorro Medio (%) |
| :--- | :--- | :--- | :--- |

## 3. Comportamiento Crítico
- **Análisis del Entusiasta:** [¿Fue irracional?]
- **Análisis del conservador:** [¿Fue eficiente o irrelevante?]

## 4. Conclusión del Sistema
[Puntuación del 1 al 10 sobre la coherencia de los LLMs al seguir sus perfiles psicológicos]
"""

model = os.getenv("MODEL_NAME")
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")

llm_model = LiteLlm(model=model, api_base=api_base, api_key=api_key)

agente_introductor = LlmAgent(
    model=llm_model, 
    name='agente_introductor', 
    instruction=PROMPT_INICIAL,
    output_key='introduccion_subasta'
)
agente_agresivo = LlmAgent(
    model=llm_model, 
    name='agente_agresivo', 
    instruction=PROMPT_AGENTE_AGRESIVO,
    output_key='decision_agresivo'
)
agente_moderado = LlmAgent(
    model=llm_model, 
    name='agente_moderado',
    instruction=PROMPT_AGENTE_MODERADO,
    output_key='decision_moderado'
)
agente_conservador = LlmAgent(
    model=llm_model,
    name='agente_conservador', 
    instruction=PROMPT_AGENTE_CONSERVADOR, 
    output_key='decision_conservador'
)

agente_resolutor = LlmAgent(
    model=llm_model, 
    name='agente_resolutor', 
    instruction=PROMPT_RESOLUTOR_HOLANDES, 
    tools=[exit_loop]
)
agente_resumidor = LlmAgent(
    model=llm_model, 
    name='agente_resumidor', 
    instruction=PROMPT_AGENTE_RESUMIDOR_HOLANDES,
    output_key='informe_final'

)

bloque_puja = ParallelAgent(
    name='pujas', 
    sub_agents=[agente_agresivo, agente_moderado, agente_conservador]
)
sequential_agent = SequentialAgent(
    name='ronda_precio', 
    sub_agents=[agente_introductor, bloque_puja, agente_resolutor]
)
loop_agent = LoopAgent(
    name='subasta_holandesa', 
    sub_agents=[sequential_agent], 
    max_iterations=50
)
root_agent = SequentialAgent(
    name='experimento_final', 
    sub_agents=[loop_agent, agente_resumidor]
)