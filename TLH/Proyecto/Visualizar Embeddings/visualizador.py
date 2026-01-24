import streamlit as st
import pandas as pd

st.set_page_config(page_title="Visor de Embeddings Pro", layout="wide")

st.title("Proyección del Espacio Latente")
st.markdown("""
Visualiza y compara agrupaciones semánticas. 
Sube tus archivos CSV (columnas requeridas: `x`, `y`, `word`, `category`. Opcional: `z`).
""")

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None
    return None

def generar_grafico(df, titulo, selected_categories, search_term, visualizacion_3d=False):
    import plotly.express as px

    req_cols = {'x', 'y', 'word', 'category'}
    if not req_cols.issubset(df.columns):
        st.error(f"⚠️ El archivo '{titulo}' no tiene las columnas requeridas: {req_cols}")
        return None

    df_plot = df.copy()

    df_filtrado = df_plot[df_plot['category'].isin(selected_categories)]

    if search_term:
        mask_category = df_plot['category'].isin(selected_categories)
        mask_search = df_plot['word'].str.contains(search_term, case=False, na=False)
        df_filtrado = df_plot[mask_category | mask_search]

    if df_filtrado.empty:
        st.warning(f"No hay datos para mostrar en {titulo} con los filtros actuales.")
        return None

    limit_text = 1500
    show_text = "word" if len(df_filtrado) < limit_text else None

    color_map = {
        'Characters': 'red', 'Entities': 'orange', 'Objects': 'blue',
        'Places': 'green', 'Verbs': 'purple', 'Others': 'gray'
    }

    if visualizacion_3d:
        if 'z' not in df_filtrado.columns:
            df_filtrado['z'] = 0
        
        fig = px.scatter_3d(
            df_filtrado, 
            x="x", y="y", z="z",
            color="category",
            text=show_text,
            hover_data=["word", "category"],
            title=f"{titulo} (3D - {len(df_filtrado)} items)",
            color_discrete_map=color_map,
            height=700
        )
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=False),
                yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=False),
                zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=False),
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(font=dict(color='white'))
        )
        
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))

        if search_term:
            found = df_filtrado[df_filtrado['word'].str.contains(search_term, case=False, na=False)]
            if not found.empty:
                fig.add_scatter3d(
                    x=found['x'], y=found['y'], z=found['z'],
                    mode='markers+text',
                    text=found['word'],
                    textfont=dict(size=14, color='yellow'),
                    marker=dict(size=15, color='yellow', opacity=0.9),
                    name='Búsqueda'
                )

    else:
        fig = px.scatter(
            df_filtrado, 
            x="x", y="y", 
            color="category",
            text=show_text,
            hover_data=["word", "category"],
            title=f"{titulo} ({len(df_filtrado)} items)",
            color_discrete_map=color_map,
            height=600,
            render_mode='webgl'
        )

        fig.update_traces(
            textposition='top center', 
            marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey'))
        )
        
        fig.update_layout(
            template="plotly_white", 
            dragmode='pan'
        )

        if search_term:
            found = df_filtrado[df_filtrado['word'].str.contains(search_term, case=False, na=False)]
            if not found.empty:
                fig.add_scatter(
                    x=found['x'], y=found['y'], 
                    mode='markers+text', 
                    text=found['word'],
                    textposition="top center",
                    textfont=dict(size=14, color='black'),
                    marker=dict(size=20, color='yellow', opacity=0.9, line=dict(width=2, color='red')),
                    name='Búsqueda'
                )

    return fig

with st.sidebar:
    st.header("📂 Datos")
    file1 = st.file_uploader("Archivo 1", type=["csv"], key="f1")
    file2 = st.file_uploader("Archivo 2", type=["csv"], key="f2")
    
    st.divider()
    
    st.header("🎨 Diseño")
    modo_dim = st.radio("Dimensiones", ["2D", "3D"], index=0, horizontal=True)
    is_3d = (modo_dim == "3D")
    
    modo_vista = st.radio(
        "Disposición", 
        ["Lado a lado", "Vertical"],
        index=0
    )
    
    st.divider()
    st.header("🔍 Filtros")
    filter_placeholder = st.container()
    search = st.text_input("Buscar término", "")



df1, df2 = None, None

if file1:
    with st.spinner('Cargando Modelo 1...'):
        df1 = load_data(file1)

if file2:
    with st.spinner('Cargando Modelo 2...'):
        df2 = load_data(file2)

all_categories = set()
if df1 is not None and 'category' in df1.columns:
    all_categories.update(df1['category'].unique())
if df2 is not None and 'category' in df2.columns:
    all_categories.update(df2['category'].unique())

if all_categories:
    with filter_placeholder:
        selected_categories = st.multiselect(
            "Categorías activas", 
            sorted(list(all_categories)),
            default=sorted(list(all_categories))
        )
else:
    selected_categories = []


if df1 is None and df2 is None:
    st.info("👈 Sube tus archivos CSV para comenzar.")
    st.stop()

def mostrar_modelo_1():
    with st.spinner('Generando gráfico 1...'):
        fig1 = generar_grafico(df1, "Modelo 1", selected_categories, search, is_3d)
        if fig1: st.plotly_chart(fig1, use_container_width=True)

def mostrar_modelo_2():
    with st.spinner('Generando gráfico 2...'):
        fig2 = generar_grafico(df2, "Modelo 2", selected_categories, search, is_3d)
        if fig2: st.plotly_chart(fig2, use_container_width=True)

if df1 is not None and df2 is not None:
    if modo_vista == "Lado a lado":
        col1, col2 = st.columns(2)
        with col1: mostrar_modelo_1()
        with col2: mostrar_modelo_2()
    else:
        mostrar_modelo_1()
        st.markdown("---")
        mostrar_modelo_2()
elif df1 is not None:
    mostrar_modelo_1()
elif df2 is not None:
    mostrar_modelo_2()