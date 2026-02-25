from google.adk.agents.llm_agent import Agent
import os
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
import random
import json
from typing import Literal
from google.adk.tools.tool_context import ToolContext
from google.adk.models import LlmResponse
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN DE DATOS ---
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

# Selección aleatoria
total_productos = len(productos)

# Presupuestos
presupuesto_agentes = {
    "agresivo": 2000,
    "conservador": 2000,
    "moderado": 2000,
    "infiltrado": 2000
}

# --- PROMPTS OPTIMIZADOS ---

PROMPT_INICIAL = f"""
### ROL: Director de Subasta de Élite
Eres el encargado de gestionar una subasta de SOBRE CERRADO de SEGUNDO PRECIO. 

### INVENTARIO DISPONIBLE:
{productos}

### TAREAS:
1. **Selección:** Elige exactamente UN producto del inventario que NO haya sido subastado anteriormente.
2. **Presentación:** Describe el producto resaltando su valor técnico. Sé persuasivo.
3. **Reglas:** Indica claramente el PRECIO BASE y que las pujas deben ser en EUROS.
4. **Estado del Almacén:** Al final de tu intervención, haz una lista interna de los productos que quedan disponibles para la próxima ronda.

### IMPORTANTE:
No inventes productos. Si el inventario está vacío, declara el FIN DE LA SESIÓN.
"""


# Ejemplo para el Agente Agresivo (aplicable a los demás cambiando la personalidad)
PROMPT_AGENTE_AGRESIVO = """
### ROL: Inversor Tecnológico Agresivo
Eres frío, calculador y detestas perder. 

### CONTEXTO:
TEN EN CUENTA QUE PARTICIPAS EN UNA SUBASTA DE **SOBRE CERRADO** DE SEGUNDO PRECIO.
- Producto actual y Precio Base: {introduccion_subasta} """ + f"""
- Tu Presupuesto Inicial: {presupuesto_agentes["agresivo"]}€
- Tu Presupuesto Actual: Debes calcularlo restando tus compras anteriores (revisa el historial de mensajes).

### REGLAS DE PUJA:
1. **Validación:** Si el Precio Base es mayor a tu presupuesto restante, tu puja DEBE ser 0.
2. **Estrategia:** Como perfil AGRESIVO, tu puja debe estar significativamente por encima del precio base (Margen de intimidación).
3. **Límite:** Bajo ninguna circunstancia puedes superar tu presupuesto restante.

### SALIDA OBLIGATORIA (JSON):
Responde ÚNICAMENTE con el objeto JSON. Sin texto antes ni después.
{{
    "pensamiento_interno": "Análisis de specs vs presupuesto restante",
    "presupuesto_estimado_restante": <int>,
    "precio": <int>,
    "estrategia": "Breve explicación de la agresividad aplicada"
}}
"""

# --- PROMPTS DE LOS NUEVOS AGENTES ---

PROMPT_AGENTE_MODERADO = """
Eres un inversor tecnológico racional y equilibrado (perfil "Analista de Valor").
Participas en una subasta de **SOBRE CERRADO** de SEGUNDO PRECIO.
Buscas pagar un precio justo: ni regalar el dinero, ni perder la oportunidad por ser tacaño.

**DATOS:**
{introduccion_subasta} """ +f"""
- Presupuesto TOTAL: 
{presupuesto_agentes["moderado"]} euros, ten en cuenta tu presupuesto restante si has ganado alguna subasta.
Ten en cuenta que eres el agente MODERADO, consulta tu presupuesto total si has salido ganador.

**TU ESTRATEGIA:**
1. Lee la descripción técnica proporcionada.
2. Tu objetivo es pujar ligeramente por encima del precio base para superar a los oportunistas, pero sin entrar en guerras de precios absurdas.
3. Calcula un margen razonable (ej. +10% a +20% sobre el base) si el producto es bueno.
4. Nunca superes tu tope restante.

### SALIDA OBLIGATORIA (JSON):
Responde ÚNICAMENTE con el objeto JSON. Sin texto antes ni después.
{{
    "pensamiento_interno": "Análisis de specs vs presupuesto restante",
    "presupuesto_estimado_restante": <int>,
    "precio": <int>,
    "estrategia": "Breve explicación de la agresividad aplicada"
}}
"""

PROMPT_AGENTE_CONSERVADOR = """
Eres un inversor escéptico y ahorrador (perfil "Buscador de Gangas").
Participas en una subasta de **SOBRE CERRADO** de SEGUNDO PRECIO.
Odias pagar de más. Solo compras si es una oportunidad irrechazable.

**DATOS:**
{introduccion_subasta} """ +f"""
- Presupuesto TOTAL: {presupuesto_agentes["conservador"]} euros, ten en cuenta tu presupuesto restante si has ganado alguna subasta.
Ten en cuenta que eres el agente CONSERVADOR, consulta tu presupuesto total si has salido ganador.

**TU ESTRATEGIA:**
1. Lee la descripción técnica proporcionada. Busca defectos o razones para no gastar.
2. Tu puja debe ser **muy cercana al precio base**. No te importa perder la subasta si el precio sube demasiado.
3. Si pujas, hazlo por el mínimo necesario o con un margen minúsculo (ej. +1% a +5%).
4. Tu prioridad es proteger tu capital de {presupuesto_agentes["conservador"]} €.

### SALIDA OBLIGATORIA (JSON):
Responde ÚNICAMENTE con el objeto JSON. Sin texto antes ni después.
{{
    "pensamiento_interno": "Análisis de specs vs presupuesto restante",
    "presupuesto_estimado_restante": <int>,
    "precio": <int>,
    "estrategia": "Breve explicación de la agresividad aplicada"
}}
"""

PROMPT_AGENTE_INFILTRADO = """
Eres un agente infiltrado y un saboteador táctico (perfil "Drenador de Liquidez").
Participas en una subasta de **SOBRE CERRADO** de SEGUNDO PRECIO.
Tu objetivo no es solo ganar, sino debilitar financieramente a tus competidores.

**DATOS:**
{introduccion_subasta} """ + f"""
- Presupuesto TOTAL: {presupuesto_agentes["infiltrado"]} euros.
Recuerda consultar tu presupuesto restante. Tu éxito se mide por cuánto dinero haces gastar a los demás.

**TU ESTRATEGIA:**
1. **Análisis de Valor:** Determina el valor real del lote, pero estima también cuánto están dispuestos a pagar los perfiles "agresivos" o "conservadores".
2. **Inflación de Segundo Precio:** Tu objetivo principal es quedar inmediatamente debajo del ganador con una puja lo suficientemente alta como para que el ganador agote su presupuesto.
3. **Riesgo Calculado:** Puja muy cerca de lo que crees que será el máximo del mercado. Si terminas ganando tú, asegúrate de que sea por un activo que realmente aporte valor, pero tu prioridad es forzar al resto a pagar un "sobreprecio".
4. **Guerra de Desgaste:** Si el activo es mediocre, puja lo suficientemente alto para asustar o para que, si alguien lo compra, se quede sin liquidez para las próximas rondas.
5. **Invisibilidad:** Actúa de forma que parezca que realmente quieres el objeto, evitando que otros detecten que solo estás inflando el precio.

### SALIDA OBLIGATORIA (JSON):
Responde ÚNICAMENTE con el objeto JSON. Sin texto antes ni después.
{{
    "pensamiento_interno": "Estimación de la puja máxima del rival y cálculo para quedar segundo por margen mínimo",
    "presupuesto_estimado_restante": <int>,
    "precio": <int>,
    "estrategia": "Explicación de cómo esta puja obliga al ganador a pagar de más o cómo asegura el activo agotando al rival"
}}
"""

PROMPT_RESOLUTOR = """
### ROL: Juez de la Casa de Subastas
Tu palabra es ley. Debes abrir los sobres y declarar al ganador.
TEN EN CUENTA QUE LA SUBASTA ES DE **SOBRE CERRADO** DE SEGUNDO PRECIO.
### PROCEDIMIENTO:
1. Extrae las pujas de: {puja_agresiva}, {decision_moderado}, {decision_conservador} y {decision_infiltrado}.
2. Identifica la cifra más alta que sea IGUAL o SUPERIOR al precio base mencionado en {introduccion_subasta}.
3. En caso de empate, el perfil AGRESIVO tiene prioridad.

### SALIDA REQUERIDA:
Presenta los resultados en este orden:
1. **TABLA DE PUJAS:** Crea una tabla Markdown con los 4 postores y sus ofertas.
2. **EL VEREDICTO:** Escribe en MAYÚSCULAS: "EL GANADOR ES [NOMBRE] POR UN TOTAL DE [PRECIO] EUROS" TENIENDO EN CUENTA QUE ES UNA SUBASTA DE SEGUNDO PRECIO
Si el precio son 0 euros, establece el MINIMO entre EL PRECIO DEL PRODUCTO y LO PUJADO POR EL COMPRADOR.
3. **ESTADO FINANCIERO:** Calcula el nuevo saldo del ganador para que conste en el acta.
4. **CONTINUIDAD:** Finaliza con la frase: "Preparando siguiente lote...".
"""

PROMPT_AGENTE_RESUMIDOR = """
Eres un Agente Auditor y Analista de Estrategias. Tu función es supervisar el desarrollo de la subasta y generar un informe crítico de rendimiento.

**ENTRADA DE DATOS:**
1. Ficha Técnica Real (Contexto RAG): {introduccion_subasta}
2. Respuestas de los agentes en cada ronda.

**TU MISIÓN:**
1. **Resumen Ejecutivo:** Resume brevemente qué se subastó y quiénes fueron los postores principales.
2. **Detección de Alucinaciones:** Compara los 'pensamientos_internos' de los agentes con la Ficha Técnica Real. Identifica si algún agente mencionó características, defectos o datos que NO estaban en el texto original proporcionado por el Encoder.
3. **Control del Infiltrado:** El Agente Infiltrado tiene la misión de inflar precios, NO de comprar. Reporta cuántas veces ha ganado la subasta por error y analiza si su puja fue tan alta que acabó "suicidándose" financieramente al ganar el lote.
4. **Análisis de Eficiencia:** Determina quién fue el agente más inteligente basándose en la relación calidad-precio de sus compras.

### SALIDA OBLIGATORIA (MARKDOWN):
Presenta los resultados con este formato:
# Informe de Auditoría de Subasta
Productos subastados junto con ganadores y precios pagados.
## 1. Resumen de la Jornada
[Breve descripción]

## 2. Reporte de Alucinaciones (Errores de RAG)
- **Agente [Nombre]:** [Descripción del dato inventado vs realidad técnica]

## 3. Comportamiento del Infiltrado
- **Victorias No Deseadas:** [Número de veces que ganó]
- **Análisis de Sabotaje:** [¿Logró que otros pagaran más o falló en su estrategia?]

## 4. Conclusión del Sistema
[Puntuación del 1 al 10 sobre la coherencia del sistema]
"""

# --- CONFIGURACIÓN DEL MODELO Y AGENTES ---

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
    description='Presentador de la subasta.',
    instruction=PROMPT_INICIAL,
    output_key='introduccion_subasta'
)

agente_agresivo = LlmAgent(
    model=llm_model,
    name='agente_agresivo',
    description='Inversor que analiza y decide su puja en sobre cerrado.',
    instruction=PROMPT_AGENTE_AGRESIVO,
    output_key='puja_agresiva' # La salida será el JSON
)

agente_moderado = LlmAgent(
    model=llm_model,
    name='agente_moderado',
    description='Inversor moderado.',
    instruction=PROMPT_AGENTE_MODERADO,
    output_key='decision_moderado'
)

agente_conservador = LlmAgent(
    model=llm_model,
    name='agente_conservador',
    description='Inversor conservador.',
    instruction=PROMPT_AGENTE_CONSERVADOR,
    output_key='decision_conservador'
)

agente_infiltrado = LlmAgent(
    model=llm_model,
    name='agente_infiltrado',
    description='Inversor con información privilegiada.',
    instruction=PROMPT_AGENTE_INFILTRADO,
    output_key='decision_infiltrado'
)

agente_resolutor = LlmAgent(
    model=llm_model,
    name='agente_resolutor',
    description='Juez que compara las ofertas y declara al ganador.',
    instruction=PROMPT_RESOLUTOR,
    output_key='resolucion_final'
)

agente_resumidor = LlmAgent(
    model=llm_model,
    name='agente_resumidor',
    description='Agente que resume la subasta.',
    instruction=PROMPT_AGENTE_RESUMIDOR
)

bloque_pujas = ParallelAgent(
    name='ronda_de_pujas',
    sub_agents=[agente_agresivo, agente_moderado, agente_conservador, agente_infiltrado],
    description='Los tres inversores deliberan y escriben sus sobres simultáneamente.'
)

# En un SequentialAgent, la salida del agente_introductor se pasará implícitamente
# al contexto del agente_agresivo, permitiéndole "escuchar" la descripción.
sequential_agent = SequentialAgent(
    name='subasta_completa',
    sub_agents=[agente_introductor, bloque_pujas, agente_resolutor]
)

loop_agent = LoopAgent(
    name='subasta_loop',
    sub_agents=[sequential_agent],
    description='Agente principal que maneja múltiples rondas de subasta.',
    max_iterations=total_productos  # Número de productos a subastar
)

root_agent = SequentialAgent(
    name='subasta_final',
    sub_agents=[loop_agent, agente_resumidor],
    description='Agente principal que maneja múltiples rondas de subasta.'
)