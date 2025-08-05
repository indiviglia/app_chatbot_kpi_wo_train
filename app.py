import streamlit as st
import os
import json
import openai
import pandas as pd
from pathlib import Path
from PIL import Image

CSV = "master_table_fixed3.csv"
PARQ = "master_volume.parquet"

# 1. Solo convierte el CSV a Parquet si aún no existe
if not Path(PARQ).exists():
    df_csv = pd.read_csv(CSV, sep=";", dtype="string")
    df_csv.to_parquet(PARQ, index=False)
    print("✅ master_volume.parquet guardado:", Path(PARQ).stat().st_size/1e6, "MB")

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Producción Farmacéutica",
    layout="wide"
)

# Función para cargar y mostrar el logo
def display_logo(location="main"):
    """Logo si existe"""
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        logo = Image.open(logo_path)
        if location == "main":
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(logo, width=300)
        elif location == "sidebar":
            st.image(logo, width=200)
        elif location == "header":
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(logo, width=100)
            with col2:
                st.title("Análisis de Producción Farmacéutica")
            return True
    return False

if not display_logo("header"):
    st.title("Análisis de Producción Farmacéutica")
st.markdown("---")

# OpenAI Client
@st.cache_resource
def init_openai_client():
    return openai.AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    )

# Load Data
@st.cache_data
def load_data():
    ART = Path("artifacts")
    df = pd.read_parquet(PARQ)
    with open(ART / "preprompt2.txt", 'r', encoding='utf-8') as f:
        preprompt = f.read()
    return df, preprompt

st.text(f"DEBUG: listo load data")

def ask_llm3_with_context(question: str, conversation_history: list, years=None):
    client = init_openai_client()
    df, preprompt = load_data()
    if years is not None:
        df = df[df["year"].astype(str).isin([str(y) for y in years])]
    system_msg = (
        preprompt
        + "\n\nUsa estos datos tabulares en CSV:\n"
        + df.head(50).to_csv(index=False)  # Limita filas si el DataFrame es grande
    )
    messages = [{"role": "system", "content": system_msg}]
    for msg in conversation_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=1000,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    welcome_message = """¡Hola! Soy tu analista de producción farmacéutica. 
    
Puedo ayudarte a analizar:
- 📊 Volúmenes de producción por año, mes, sustancia o línea
- 💊 Rankings de sustancias y presentaciones
- 📈 Tendencias y picos de producción
- 🏭 Análisis por líneas de producción

¿En qué puedo ayudarte hoy?"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz una pregunta sobre los datos de producción..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analizando datos..."):
            try:
                response = ask_llm3_with_context(prompt, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")

with st.sidebar:
    display_logo("sidebar")
    st.header("Información")
    st.markdown("""
    ### Datos disponibles:
    - 📅 Periodo: 2024 (año completo) y 2025 (hasta mayo inclusive)
    - 🏭 Planta: Laboratorios Liconsa S.A.
    - 💊 700+ presentaciones comerciales
    - 🧪 90+ sustancias activas
    - 🏭 Múltiples líneas de producción
    
    ### Ejemplos de solicitudes:
        - Elabora un top 10 de insights de negocio 
          que sirvan para la toma de decisiones. 
        -⁠ ¿Qué presentación comercial tuvo mayor impacto global en 2025 y 
          con qué % de importancia? Explícame qué sería “impacto global”.
        -⁠ ¿Cuáles son las top 5 sustancias por volumen?
        - ¿Qué familia fue la que generó la mayor variación de volumen en 
          octubre 2024 y qué presentación comercial fue de esa familia 
          la más utilizada?.
        -⁠ ⁠¿Cómo evolucionó la producción de OMEPRAZOL?
    """)
    st.markdown("---")
    st.markdown(f"**💬 Mensajes en contexto:** {len(st.session_state.messages)}")
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()
