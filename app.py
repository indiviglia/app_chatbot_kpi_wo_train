import streamlit as st
import os
import json
import openai
import pandas as pd
from pathlib import Path
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Producción Farmacéutica [SIN ENTRENAMIENTO]",
    layout="wide"
)

# Función para cargar y mostrar el logo
def display_logo(location="main"):
    logo_path = Path("assets/logo.png")
    if not logo_path.exists():
        return False
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
            st.title("Análisis de Producción Farmacéutica [SIN ENTRENAMIENTO]")
        return True
    return False

if not display_logo("header"):
    st.title("Análisis de Producción Farmacéutica [SIN ENTRENAMIENTO]")
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
    # 1) Determinar carpeta base
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()
    ART = base_dir / "artifacts"

    # 2) Verificar presencia de archivos
    csv_file = ART / "master_table_fixed3.csv"
    prompt_file = ART / "preprompt2.txt"
    if not csv_file.exists() or not prompt_file.exists():
        st.error("❌ Faltan archivos en artifacts/")
        return {}, ""

    # 3) Leer CSV y parsear fecha
    df = pd.read_csv(csv_file, sep=";")
    df["order_process_start_dt"] = pd.to_datetime(df["order_process_start_dt"])
    
     try:
        df = pd.read_csv(csv_file, sep=";")
    except pd.errors.EmptyDataError:
        st.error("❌ El CSV está vacío o malformado.")
        return {}, ""
        
    # 4) Calcular campos
    df["year"]   = df["order_process_start_dt"].dt.year
    df["period"] = df["order_process_start_dt"].dt.to_period("M")
    df = df.sort_values("period")
    df["lag1"]   = df["volumen_final"].shift(1)
    df["lag2"]   = df["volumen_final"].shift(2)
    df["ma3"]    = df["volumen_final"].rolling(3).mean()
    df["month"]  = df["period"].dt.month
    df["quarter"]= df["period"].dt.quarter
    df["fase_new"]= (df["year"] >= 2023).astype(int)

    # 5) Generar payloads por año
    payloads = {
        str(yr): df[df["year"] == yr].to_dict(orient="records")
        for yr in sorted(df["year"].unique())
    }

    # 6) Leer preprompt
    preprompt = prompt_file.read_text(encoding="utf-8")
    return payloads, preprompt

# Función para llamar al LLM con contexto
def ask_llm3_with_context(question: str, history: list, years=None):
    client = init_openai_client()
    payloads, preprompt = load_data()
    years = years or list(payloads.keys())
    data = {yr: payloads[yr] for yr in years if yr in payloads}

    system_msg = preprompt + "\n\nDATOS (JSON por año):\n" + json.dumps(data, ensure_ascii=False, indent=2)
    messages = [{"role": "system", "content": system_msg}] + history[-20:]
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=1000,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """¡Hola! Soy tu analista de producción farmacéutica.
Puedo ayudarte a analizar volúmenes, tendencias y KPIs de producción."""
    }]

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Haz una pregunta sobre los datos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                answer = ask_llm3_with_context(prompt, st.session_state.messages)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

