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

    # 4) Calcular campos
    df["year"]   = df["order_process_start_dt"].dt.year
    df["period"] = df["order_process_start_dt"].dt.to_period("M]()
