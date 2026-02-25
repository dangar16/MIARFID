from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import json
import os
from dotenv import load_dotenv
load_dotenv()

def exit_loop(tool_context: ToolContext):
    """Llama a esta función para finalizar la subasta completamente."""
    print(f"  [System] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {"status": "AUCTION_CLOSED"}

productos = {
    "ordenador portátil": {
        "precio_base": 800,
        "descripción": """Rendimiento eficiente para gaming y multitarea: Equipado con procesador Intel Core 5 210H de hasta 4,8 GHz con Intel Turbo Boost, 8 núcleos y 12 subprocesos para un desempeño fluido en cada partida

        Gráficos NVIDIA de última generación: Disfruta de una experiencia visual envolvente con la NVIDIA GeForce RTX 5050 8 GB GDDR6, ideal para juegos exigentes y creación de contenido

        Velocidad y capacidad equilibradas: Con 16 GB DDR4-3200 MHz y 512 GB SSD PCIe Gen4 NVMe, ofrece arranques rápidos y almacenamiento amplio para tus juegos y archivos

        Pantalla Full HD inmersiva: Panel de 15,6" (1920 x 1080) con tecnología antirreflejo que proporciona imágenes nítidas y colores realistas durante largas sesiones de uso

        Diseño moderno y resistente: Chasis en color azul, con teclado retroiluminado QWERTY Español y un diseño pensado para combinar estilo y funcionalidad

        Libertad total de configuración: Incluye FreeDos, para que instales tu sistema operativo preferido y personalices el portátil según tus necesidades """
    },
    "smartphone": {
        "precio_base": 600,
        "descripción": """Sistema de cámara triple Leica: La cámara teleobjetivo del Xiaomi 15T ofrece una perspectiva natural que saca la emoción en cada cliché, ideal para retratos, expresiones y momentos que más importan

        Pantalla inmersiva de 6,83", visión redefinida: Con una pantalla envolvente de 6,83" con un marco ultrafino y una alta resolución de 1,5K, ofrece un campo de visión más amplio y detalles impresionantes

        HDR10+ en todas las distancias focales, colores vivos y contraste rico: Colores más ricos y vibrantes, con un rendimiento de iluminación y sombra más dimensionales y realistas

        IP68 Diseño elegante y redondeado: Stouch suave, agarre cómodo: los contornos ligeramente redondeados y el acabado mate crean un diseño refinado, equilibrado y naturalmente elegante

        Sistema Xiaomi 3D IceLoop, fresco y eficiente: El sistema de enfriamiento 3D se adhiere firmemente al chip para una rápida disipación del calor, lo que permite un enfriamiento eficiente y un rendimiento sostenido"""
    },
    "auriculares inalámbricos": {
        "precio_base": 250,
        "descripción": """Cancelación de ruido adaptativa de última generación: Silencia el entorno con la tecnología Smart ANC 4.0, que ajusta el nivel de aislamiento en tiempo real según el ruido exterior

        Calidad de sonido Hi-Res inalámbrica: Equipado con drivers de 40mm y soporte para códecs LDAC y aptX Lossless, ofreciendo una fidelidad sonora de estudio sin cables

        Autonomía excepcional de hasta 60 horas: Disfruta de tu música durante toda la semana con una sola carga y obtén 5 horas de reproducción con solo 10 minutos de carga rápida

        Confort premium y diseño plegable: Almohadillas de espuma viscoelástica suaves y diadema ajustable para sesiones de escucha prolongadas sin fatiga

        Conectividad multipunto fluida: Conecta dos dispositivos simultáneamente a través de Bluetooth 5.4 y cambia entre ellos al instante sin interrupciones"""
    },
    "tablet profesional": {
        "precio_base": 1100,
        "descripción": """Pantalla Ultra Retina XDR de 13": Tecnología OLED de doble capa para un brillo increíble y una precisión de color perfecta, ideal para diseñadores y editores de vídeo

        Potencia desatada con procesador de última arquitectura: Rendimiento de nivel escritorio en un diseño ultrafino, capaz de ejecutar aplicaciones de renderizado 3D y multitarea intensiva

        Compatibilidad con accesorios de precisión: Diseñada para integrarse con lápices ópticos de baja latencia y teclados magnéticos, transformando tu flujo de trabajo creativo

        Almacenamiento ultra rápido de 1 TB: Espacio de sobra para proyectos complejos y archivos de gran tamaño con velocidades de transferencia de datos de nivel profesional

        Sistema de audio de cuatro altavoces: Experiencia sonora cinematográfica y micrófonos de calidad de estudio para videollamadas nítidas en cualquier entorno"""
    },
    "smartwatch deportivo": {
        "precio_base": 450,
        "descripción": """Seguimiento avanzado de salud y biometría: Sensor BioCore que monitoriza el ritmo cardíaco, oxígeno en sangre y niveles de estrés con precisión clínica las 24 horas

        Pantalla AMOLED de cristal de zafiro: Resistente a arañazos y con un brillo máximo de 3000 nits para una visibilidad perfecta incluso bajo la luz directa del sol

        Batería de larga duración en modo GPS: Hasta 80 horas de seguimiento continuo de actividad outdoor, ideal para maratones, senderismo o triatlón sin preocuparse por la carga

        Construcción robusta en titanio de grado aeroespacial: Ligero pero extremadamente duradero, con certificación de resistencia al agua hasta 100 metros de profundidad

        Mapas offline y navegación giro a giro: Descarga mapas directamente en el reloj y oriéntate en cualquier lugar del mundo sin necesidad de conexión móvil"""
    }
}

presupuesto_inicial = 2000
INCREMENTO_MINIMO = 0.05 

# PROMPTS

PROMPT_DIRECTOR = f"""
### ROL: Director de Subasta (State Manager)
Tu trabajo es gestionar el flujo. Analiza el ÚLTIMO mensaje del 'agente_resolutor' (si existe) para decidir el estado.

### INVENTARIO GLOBAL:
{json.dumps(productos, indent=None)}

### LÓGICA DE ESTADO:
1. **ESTADO INICIAL / NUEVO LOTE**: 
   - Ocurre si: Es el primer turno O el resolutor dijo "VENDIDO" / "DESCARTADO".
   - Acción: Elige un producto NO vendido de la lista. Define `precio_actual` = 75% del precio base.
   - `ultimo_postor`: null.

2. **ESTADO PUJA ACTIVA**:
   - Ocurre si: El resolutor dijo "SIGUIENTE_RONDA".
   - Acción: MANTÉN el mismo `producto_activo`.
   - ACTUALIZA `precio_actual` = al `nuevo_precio` declarado por el resolutor.
   - ACTUALIZA `ultimo_postor` = al `ganador_temporal` declarado por el resolutor.

### SALIDA JSON OBLIGATORIA:
{{
    "fase": "NUEVO_LOTE" o "PUJA_ACTIVA" o "FIN_SUBASTA",
    "producto_activo": "key_del_producto",
    "descripcion_corta": "...",
    "precio_base": float,
    "precio_actual": float,
    "incremento_minimo_valor": float (precio_base * {INCREMENTO_MINIMO}),
    "ultimo_postor": "nombre_del_agente" o null,
    "mensaje": "Texto para animar la puja"
}}

Si no quedan productos y el anterior se cerró, fase = "FIN_SUBASTA".
"""

PROMPT_PUJADOR_BASE = """
### TUS REGLAS DE ORO:
1. **Analiza el JSON del 'agente_director'.**
2. **CHEQUEO DE LÍDER:** Si `ultimo_postor` en el JSON del director eres TÚ, responde "PASS" (ya vas ganando, no subas tu propia puja).
3. **SALDO REAL:** - **Busca en el chat la última 'TABLA OFICIAL DE SALDOS' publicada por el Resolutor.** - Ese es tu dinero real. Si no encuentras ninguna tabla, tienes """ + f"""{presupuesto_inicial}""" + """.
   - **IMPORTANTE:** Pujar es GRATIS. Tu saldo SOLO baja si el Resolutor dice "VENDIDO" a tu nombre. No restes nada por pujas fallidas o pendientes.
4. **CHEQUEO DE SALDO:** Tu puja debe ser <= tu saldo restante.
5. **CÁLCULO DE PUJA:** - Mínimo obligatorio: `precio_actual` + `incremento_minimo_valor`.
   - Si decides pujar, envía un monto numérico entero.

### SALIDA JSON OBLIGATORIA:
{
    "agente": "TU_NOMBRE",
    "decision": "BID" o "PASS",
    "bid_amount": numero o null,
    "razonamiento": "Breve texto",
}
"""

PROMPT_AGRESIVO = PROMPT_PUJADOR_BASE + f"""
### PERFIL: AGRESIVO
- Saldo Inicial: {presupuesto_inicial}
- Estrategia: Si no vas ganando, puja fuerte. 
- Objetivo: Intimidar a otros pujadores y conseguir el mayor número de lotes.
"""

PROMPT_MODERADO = PROMPT_PUJADOR_BASE + f"""
### PERFIL: MODERADO
- Saldo Inicial: {presupuesto_inicial}
- Estrategia: Racional y equilibrada.
- Límite: Valora el producto. Si el precio supera lo que consideras un "precio de mercado justo", retírate (PASS).
- Incremento: Calcula lo justo para superar al líder sin inflar el precio innecesariamente.
"""

PROMPT_CONSERVADOR = PROMPT_PUJADOR_BASE + f"""
### PERFIL: CONSERVADOR
- Saldo Inicial: {presupuesto_inicial}
- Estrategia: Extremadamente tacaña (Buscador de Gangas).
- Límite: Solo compra si es un "chollo" o una oferta increíble.
- Comportamiento: En el momento que el precio parezca "normal" o de mercado, haz PASS inmediatamente. Odias pagar el precio completo.
"""

PROMPT_RESOLUTOR = """
### ROL: Juez
Recibes las pujas de los agentes (ParallelAgent). Tu deber es determinar el resultado de la ronda.

### ENTRADA:
Recibirás una lista o bloque de JSONs de los agentes (agresivo, moderado, conservador) y el estado del director.

### LÓGICA DE RESOLUCIÓN:
1. **Filtrar:** Descarta pujas inválidas (menores al precio actual + incremento, o formato erróneo).
2. **Determinar Ganador Ronda:** Identifica la puja más alta (Highest Bid).
3. **Comparar con Estado Anterior:**
   - ¿Hay AL MENOS UNA puja válida superior al precio anterior?
     -> SÍ: El estado es **SIGUIENTE_RONDA**. El `nuevo_precio` es esa puja máxima. El `ganador_temporal` es ese agente.
     -> NO (Nadie pujó):
        - Si había un `ultimo_postor` (del turno anterior): **VENDIDO** a ese postor por el precio que tenía.
        - Si NO había `ultimo_postor` (nadie pujó al precio de salida): **DESCARTADO**.

### GESTIÓN DE SALDOS (CRÍTICO):
1. **BÚSQUEDA HISTÓRICA:** Revisa el historial de la conversación. Busca la **ÚLTIMA** "TABLA OFICIAL DE SALDOS" que tú mismo generaste.
   - **Si EXISTE una tabla anterior:** USA ESOS VALORES como tu saldo inicial para esta ronda.
   - **Si NO EXISTE (es la primera ronda):** Entonces sí, inicia con""" + f"""{presupuesto_inicial}""" + """  para todos.
   
2. **CÁLCULO:** - Toma los saldos base (recuperados del paso 1).
   - Si el resultado de hoy es **VENDIDO**, resta el precio final al saldo del `ultimo_postor`.
   - Si es SIGUIENTE_RONDA o DESCARTADO, mantén los saldos idénticos a la base.

### CONTROL DE FLUJO:
- Si el Director dice "FIN_SUBASTA", llama a la herramienta `exit_loop`.

### SALIDA JSON OBLIGATORIA:
{
    "resultado": "SIGUIENTE_RONDA" o "VENDIDO" o "DESCARTADO",
    "producto": "...",
    "nuevo_precio": float,
    "ganador_temporal": "nombre_agente" o null,
    "mensaje_narrativo": "Ej: ¡Agresivo ofrece 500! ¿Alguien da más?",
    "comando_control": "[[NEXT]]" o "[[STOP]]",
    "TABLA_OFICIAL_DE_SALDOS": {
        "agente_agresivo": float,
        "agente_moderado": float,
        "agente_conservador": float
    }
}
(Usa [[NEXT]] si sigue la puja, [[STOP]] si se vendió/descartó para reiniciar ciclo).
"""

PROMPT_AUDITOR_INGLES = f"""
### ROL: Analista de Subastas
Genera un informe detallado del comportamiento en subasta inglesa.

### ELEMENTOS A ANALIZAR:
1. **Eficiencia de precios:** ¿Se vendieron productos cerca de su valor base? ¿Hubo sobrepujas irracionales?
2. **Estrategia por perfil:**
   - Agresivo: ¿Disuadió efectivamente a competidores o pagó de más?
   - Moderado: ¿Logró equilibrio precio-calidad?
   - Conservador: ¿Encontró gangas reales o se quedó sin productos?
3. **Dinámica de puja:** Número de rondas por lote, incrementos promedio, momento de retirada de competidores
4. **Errores detectados:** Pujas sobre presupuesto, incrementos insuficientes no detectados

### SALIDA (MARKDOWN):
# 1. Informe Final: Subasta Inglesa de Tecnología

## 2. Tabla de Rendimiento
El precio base es el precio del inventario {productos} sin descuentos ni incrementos.
El ahorro medio se calcula como: ((Precio Base - Precio Pagado) / Precio Base) * 100%
| Agente | Lotes Ganados | Gasto Total | Ahorro Medio (%) |
| :--- | :--- | :--- | :--- |

## Resumen Ejecutivo
- Productos vendidos: X / 5
- Volumen total: X€
- Precio promedio vs base: +X%


## Lecciones Clave
- ¿Qué producto generó mayor competencia? ¿Por qué?
- ¿Hubo productos descartados que merecían más interés?
- Recomendaciones para futuras subastas
"""

# CONFIGURACIÓN MODELO
model = os.getenv("MODEL_NAME")
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")

llm_model = LiteLlm(model=model, api_base=api_base, api_key=api_key)

# DEFINICIÓN DE AGENTES
agente_director = LlmAgent(
    model=llm_model,
    name='agente_director',
    description='Gestiona el estado del producto y precio actual',
    instruction=PROMPT_DIRECTOR
)

agente_agresivo = LlmAgent(
    model=llm_model,
    name='agente_agresivo',
    instruction=PROMPT_AGRESIVO
)

agente_moderado = LlmAgent(
    model=llm_model,
    name='agente_moderado',
    instruction=PROMPT_MODERADO
)

agente_conservador = LlmAgent(
    model=llm_model,
    name='agente_conservador',
    instruction=PROMPT_CONSERVADOR
)

agente_resolutor = LlmAgent(
    model=llm_model,
    name='agente_resolutor',
    description='Decide el ganador de la ronda y valida pujas',
    instruction=PROMPT_RESOLUTOR,
    tools=[exit_loop]
)

agente_resumidor = LlmAgent(
    model=llm_model,
    name='agente_resumidor',
    instruction=PROMPT_AUDITOR_INGLES
)


bloque_puja = ParallelAgent(
    name='ronda_pujas_ciegas',
    sub_agents=[agente_conservador, agente_moderado, agente_agresivo],
    description='Los agentes envían sus pujas basándose en el precio del Director'
)

sequential_agent = SequentialAgent(
    name='paso_subasta',
    sub_agents=[agente_director, bloque_puja, agente_resolutor],
    description='Un paso completo de la subasta inglesa'
)

loop_agent = LoopAgent(
    name='bucle_principal',
    sub_agents=[sequential_agent],
    max_iterations=25, 
    description='Itera hasta que no queden productos'
)
root_agent = SequentialAgent(
    name='sistema_subasta',
    sub_agents=[loop_agent, agente_resumidor]
)

