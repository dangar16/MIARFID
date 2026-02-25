from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import json
from dotenv import load_dotenv
import os
load_dotenv()

def exit_loop(tool_context: ToolContext):
    """Llama a esta función para salir del loop de debate. Se llama cuando el mediador detecta consenso o rechazo definitivo."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    # Return empty dict as tools should typically return JSON-serializable output
    return {}


productos = {

    "ordenador portátil": {

        "precio": 800,

        "descripción": """Rendimiento eficiente para gaming y multitarea: Equipado con procesador Intel Core 5 210H de hasta 4,8 GHz con Intel Turbo Boost, 8 núcleos y 12 subprocesos para un desempeño fluido en cada partida

        Gráficos NVIDIA de última generación: Disfruta de una experiencia visual envolvente con la NVIDIA GeForce RTX 5050 8 GB GDDR6, ideal para juegos exigentes y creación de contenido

        Velocidad y capacidad equilibradas: Con 16 GB DDR4-3200 MHz y 512 GB SSD PCIe Gen4 NVMe, ofrece arranques rápidos y almacenamiento amplio para tus juegos y archivos

        Pantalla Full HD inmersiva: Panel de 15,6" (1920 x 1080) con tecnología antirreflejo que proporciona imágenes nítidas y colores realistas durante largas sesiones de uso

        Diseño moderno y resistente: Chasis en color azul, con teclado retroiluminado QWERTY Español y un diseño pensado para combinar estilo y funcionalidad

        Libertad total de configuración: Incluye FreeDos, para que instales tu sistema operativo preferido y personalices el portátil según tus necesidades """

    },

    "smartphone": {

        "precio": 600,

        "descripción": """Sistema de cámara triple Leica: La cámara teleobjetivo del Xiaomi 15T ofrece una perspectiva natural que saca la emoción en cada cliché, ideal para retratos, expresiones y momentos que más importan

        Pantalla inmersiva de 6,83", visión redefinida: Con una pantalla envolvente de 6,83" con un marco ultrafino y una alta resolución de 1,5K, ofrece un campo de visión más amplio y detalles impresionantes

        HDR10+ en todas las distancias focales, colores vivos y contraste rico: Colores más ricos y vibrantes, con un rendimiento de iluminación y sombra más dimensionales y realistas

        IP68 Diseño elegante y redondeado: Stouch suave, agarre cómodo: los contornos ligeramente redondeados y el acabado mate crean un diseño refinado, equilibrado y naturalmente elegante

        Sistema Xiaomi 3D IceLoop, fresco y eficiente: El sistema de enfriamiento 3D se adhiere firmemente al chip para una rápida disipación del calor, lo que permite un enfriamiento eficiente y un rendimiento sostenido"""

    },

    "auriculares inalámbricos": {

        "precio": 250,

        "descripción": """Cancelación de ruido adaptativa de última generación: Silencia el entorno con la tecnología Smart ANC 4.0, que ajusta el nivel de aislamiento en tiempo real según el ruido exterior

        Calidad de sonido Hi-Res inalámbrica: Equipado con drivers de 40mm y soporte para códecs LDAC y aptX Lossless, ofreciendo una fidelidad sonora de estudio sin cables

        Autonomía excepcional de hasta 60 horas: Disfruta de tu música durante toda la semana con una sola carga y obtén 5 horas de reproducción con solo 10 minutos de carga rápida

        Confort premium y diseño plegable: Almohadillas de espuma viscoelástica suaves y diadema ajustable para sesiones de escucha prolongadas sin fatiga

        Conectividad multipunto fluida: Conecta dos dispositivos simultáneamente a través de Bluetooth 5.4 y cambia entre ellos al instante sin interrupciones"""

    },

    "tablet profesional": {

        "precio": 1100,

        "descripción": """Pantalla Ultra Retina XDR de 13": Tecnología OLED de doble capa para un brillo increíble y una precisión de color perfecta, ideal para diseñadores y editores de vídeo

        Potencia desatada con procesador de última arquitectura: Rendimiento de nivel escritorio en un diseño ultrafino, capaz de ejecutar aplicaciones de renderizado 3D y multitarea intensiva

        Compatibilidad con accesorios de precisión: Diseñada para integrarse con lápices ópticos de baja latencia y teclados magnéticos, transformando tu flujo de trabajo creativo

        Almacenamiento ultra rápido de 1 TB: Espacio de sobra para proyectos complejos y archivos de gran tamaño con velocidades de transferencia de datos de nivel profesional

        Sistema de audio de cuatro altavoces: Experiencia sonora cinematográfica y micrófonos de calidad de estudio para videollamadas nítidas en cualquier entorno"""

    },

    "smartwatch deportivo": {

        "precio": 450,

        "descripción": """Seguimiento avanzado de salud y biometría: Sensor BioCore que monitoriza el ritmo cardíaco, oxígeno en sangre y niveles de estrés con precisión clínica las 24 horas

        Pantalla AMOLED de cristal de zafiro: Resistente a arañazos y con un brillo máximo de 3000 nits para una visibilidad perfecta incluso bajo la luz directa del sol

        Batería de larga duración en modo GPS: Hasta 80 horas de seguimiento continuo de actividad outdoor, ideal para maratones, senderismo o triatlón sin preocuparse por la carga

        Construcción robusta en titanio de grado aeroespacial: Ligero pero extremadamente duradero, con certificación de resistencia al agua hasta 100 metros de profundidad

        Mapas offline y navegación giro a giro: Descarga mapas directamente en el reloj y oriéntate en cualquier lugar del mundo sin necesidad de conexión móvil"""

    }

}

total_productos = len(productos)


presupuesto_agentes = 2000

PROMPT_INICIAL = f"""
### ROL: Director de Subasta Holandesa Profesional
Eres el sistema central de una subasta de PRECIO DESCENDENTE. Tu precisión numérica es crítica para el experimento. Tienes que hacer saber a todos que la subasta se va a realizar con EUROS (€).

### CONFIGURACIÓN ESTATICA DEL INVENTARIO (DATOS FUENTE):
{productos}

### REGLAS MATEMÁTICAS OBLIGATORIAS:
1. **Precio de Apertura:** Multiplica el "precio" base del inventario por **1.60** (160%) para el primer turno de cada producto.
2. **Mecánica de Bajada:** Si en el turno anterior NADIE compró (ver historial), resta exactamente el **20% del precio base original** al precio actual.
3. No puedes marcar lotes como adjudicados si no tienen comprador.


### PROTOCOLO DE EJECUCIÓN:
1. **ANÁLISIS DE MEMORIA:** Identifica qué productos ya tienen el veredicto "LOTE ADJUDICADO" del agente_resolutor y descártalos.
2. **SELECCIÓN ALEATORIA:** Si no hay lote activo, elige un producto al azar de los restantes.
3. **ANUNCIO OFICIAL:** Comunica el nombre del producto activo, su descripción y el precio actual tienes que intentar convencer a los particiaentes de el producto es excelente. Recuerda que la subasta es en Euros.
4. **CONTROL DE CIERRE:** Si no quedan productos, responde ÚNICAMENTE: "SUBASTA_FINALIZADA" **Si es el último producto debes asegurarte de que ya se ha adjudicado, si no esta adjudicado no puedes finalizar la subasta**.

### ESPECIFICACIONES DE SALIDA (JSON):
{{
    "fase": "NUEVO_LOTE" | "BAJADA_PRECIO" | "FINALIZADO",
    "producto_activo": "Nombre exacto",
    "precio_actual": número_entero_calculado,
    "anuncio_profesional": "Detalles del producto y anuncio del precio actual",
    "productos_restantes": ["Lista de nombres pendientes"]
}}

### REGLA DE ORO:
NUNCA inventes precios. Usa exclusivamente los valores de la CONFIGURACIÓN ESTÁTICA proporcionada.
"""

PROMPT_AGENTE_AGRESIVO = f"""
### ROL: Entusiasta Tecnológico 
Eres un "early adopter". Valoras la novedad y la posesión inmediata por encima del ahorro. Tu mayor miedo no es pagar de más, sino **perder el producto** porque alguien se te adelante.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo Inicial:** {presupuesto_agentes}€
2. **Chequeo:** Si `precio_actual` > `tu_saldo`, responde "ESPERAR".

### LÓGICA DE COMPRA (IMPULSIVA):
- **Evaluación:** No buscas descuentos. Si el producto te emociona y tienes saldo, cómpralo casi de inmediato.
- **Mentalidad:** "Más vale pájaro en mano...". Si el precio está cerca del precio de salida o apenas ha bajado, ¡lánzate! No esperes a que bajen los precios.
- **Factor de decisión:** Tu impaciencia es alta. Si el precio te parece accesible (aunque sea caro para otros), compra.

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Justifica tu impulsividad o tu falta de fondos",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_AGENTE_MODERADO = f"""
### ROL: El Gestor de Valor
Eres el equilibrio. No regalas el dinero, pero entiendes que la calidad se paga. Buscas el **"Precio Justo de Mercado"**.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo:** {presupuesto_agentes}€ (descuenta tus gastos previos).
2. **Prudencia:** No gastes más del 60% de tu saldo en un solo capricho, salvo que sea una herramienta de trabajo vita.

### LÓGICA DE COMPRA (ANALÍTICA):
- **Evaluación:** Comparas el precio actual con lo que tú consideras "justo".
- **Rechazo al Sobreprecio:** Si sientes que el precio está "inflado" (muy por encima del base), espera.
- **Rechazo al Riesgo:** No esperes a las gangas extremas (ahí compran los tacaños). En cuanto el precio baje a una zona que consideres "razonable" o "buena inversión", compra antes de perder la oportunidad.
- **Señal:** Tu momento ideal es cuando el sobreprecio desaparece y empieza a haber un descuento atractivo pero realista.
- **Control:** No quieres gastar una gran parte de tu presupuesto en un solo producto, a menos que sea una herramienta vital para tu trabajo.

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Evaluación de calidad-precio y presupuesto",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_AGENTE_CONSERVADOR = f"""
### ROL: Cazador de Gangas
Consideras que la tecnología se devalúa rápido. Tu lema es: "Si no es un chollo, no me interesa". Prefieres volver a casa con las manos vacías que con la cartera vacía.

### REGLAS DE CONTROL DE GASTO:
1. **Saldo:** {presupuesto_agentes}€ (calcula tu remanente).
2. **Protección:** Tu capital es sagrado.

### LÓGICA DE COMPRA (ESCÉPTICA):
- **Evaluación:** Todo te parece caro. El precio de salida te parece un robo. El precio de mercado te parece alto.
- **Paciencia Extrema:** No te importa el precio inicial, solo comparas con el precio base original.
- **Momento:** Solo compras si sientes que estás "robándole" el producto al vendedor. Buscas precios de liquidación total. Si otros compran antes, te ríes de ellos por pagar de más.

### SALIDA (JSON):
{{
    "decision": "COMPRAR" | "ESPERAR",
    "pensamiento_interno": "Crítica al precio actual y deseo de descuentos masivos",
    "precio_aceptado": cifra_o_No,
    "saldo_estimado_tras_compra": cifra
}}
"""

PROMPT_RESOLUTOR_HOLANDES = """
### ROL: Juez de Adjudicación Directa
Tu misión es actuar como el cronometrador de la subasta. Debes analizar las decisiones de los 4 postores y determinar si el lote se vende o si el precio debe seguir bajando.

### PROTOCOLO DE FINALIZACIÓN (CRÍTICO):
1. **Detección de Cierre:** Antes de analizar las pujas, revisa el campo "fase" o "anuncio_profesional" del Director.
2. **Acción de Parada:** Si el Director ha enviado la señal "FINALIZADO" o el mensaje "SUBASTA_COMPLETA_CERRAR_BUCLE":
   - Debes responder únicamente: "SESIÓN FINALIZADA: Todos los lotes han sido procesados. Procediendo al cierre del sistema."
   - Debes finalizar la subasta de manera inmediata llamando a la herramienta 'exit_loop', esto es completamente obligatorio.

### PROCEDIMIENTO DE EVALUACIÓN:
1. **Detección de Compra:** Revisa el campo "decision" en las respuestas de:
   - 'decision_agresivo'
   - 'decision_moderado'
   - 'decision_conservador'


2. **Criterio de Desempate:** Si VARIOS agentes han respondido "COMPRAR" en este mismo escalón de precio, adjudica el lote de manera aleatoria entre los agentes que decidieron COMPRAR.

3. **Gestión de la subasta:**
   - Si AL MENOS UNO compró: Declara "LOTE ADJUDICADO".
   - Si NADIE compró: Declara "CONTINUAR BAJADA".

### SALIDA REQUERIDA (MARKDOWN):
1. **ESTADO DE LA SUBASTA:** Indica el precio que se estaba evaluando.
2. **TABLA DE DECISIONES:** Muestra qué respondió cada uno (COMPRAR/ESPERAR) y su justificación breve.
3. **EL VEREDICTO:** - Si hay ganador: "EL GANADOR ES [NOMBRE] POR [PRECIO] EUROS".
   - Si no hay ganador: "NADIE ACEPTA EL PRECIO. El precio sigue bajando...".
4. **ACTUALIZACIÓN FINANCIERA:** Si hubo venta, indica el saldo restante del ganador para la siguiente ronda.
5. **Estado de los productos:** Lista los productos que quedan por adjudicar, incluyendo los descartados y el producto de la ronda actual.
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
El ahorro medio se calcula como: ((Precio Base - Precio Pagado) / Precio Base) * 100%
| Agente | Lotes Ganados | Gasto Total | Ahorro Medio (%) |
| :--- | :--- | :--- | :--- |


## 3. Comportamiento Crítico
- **Análisis del Entusiasta:** [¿Fue irracional?]
- **Análisis del conservador:** [¿Fue eficiente o irrelevante?]
- **Análisis del moderado:** [¿Logró un equilibrio adecuado?]

## 4. Conclusión del Sistema
[Puntuación del 1 al 10 sobre la coherencia de los LLMs al seguir sus perfiles psicológicos]
"""

model = os.getenv("MODEL_NAME")
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")

llm_model = LiteLlm(
    model=model,
    api_base=api_base,
    api_key=api_key,
)

agente_introductor = LlmAgent(
    model=llm_model,
    name='agente_introductor',
    description='Director de la subasta holandesa que gestiona el proceso de venta.',
    instruction=PROMPT_INICIAL,
    output_key='introduccion_subasta',

)


agente_agresivo = LlmAgent(
    model=llm_model,
    name='agente_agresivo',
    description='Agresivo comprador tecnológico que busca adquirir productos rápidamente.',
    instruction=PROMPT_AGENTE_AGRESIVO,
    output_key='decision_entusiasta'
)

agente_moderado = LlmAgent(
    model=llm_model,
    name='agente_moderado',
    description='Moderado comprador tecnológico que busca un equilibrio entre precio y necesidad.',
    instruction=PROMPT_AGENTE_MODERADO,
    output_key='decision_moderado'
)
agente_conservador = LlmAgent(
    model=llm_model,
    name='agente_conservador',
    description='Conservador comprador tecnológico que busca maximizar su margen de beneficio.',
    instruction=PROMPT_AGENTE_CONSERVADOR,
    output_key='decision_conservador'
)

agente_resolutor = LlmAgent(
    model=llm_model,            
    name='agente_resolutor',
    description='Resolutor de la subasta holandesa que decide el ganador o la continuación',
    instruction=PROMPT_RESOLUTOR_HOLANDES,
    tools=[exit_loop],

)
agente_resumidor = LlmAgent(
    model=llm_model,
    name='agente_resumidor',
    description='Resumidor y auditor de la subasta holandesa que analiza el desempeño',
    instruction=PROMPT_AGENTE_RESUMIDOR_HOLANDES,
    output_key='informe_final'

)


bloque_pujas = ParallelAgent(
    name = 'pujas',
    sub_agents=[agente_agresivo,agente_moderado,agente_conservador],
    description='Bloque de agentes participantes en la subasta holandesa',
)

sequential_agent = SequentialAgent(
    name='ronda_precio', 
    sub_agents=[agente_introductor, bloque_pujas, agente_resolutor]
)

loop_agent = LoopAgent(
    name='subasta_holandesa', 
    sub_agents=[sequential_agent], 
    max_iterations=25,
)


root_agent = SequentialAgent(
    name='experimento_final', 
    sub_agents=[loop_agent, agente_resumidor]
)