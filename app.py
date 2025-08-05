import streamlit as st
import os
import json
import openai
import pandas as pd
from pathlib import Path
from PIL import Image

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
            # Opción 1: Logo centrado en el header principal
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(logo, width=300)
        
        elif location == "sidebar":
            # Opción 2: Logo en el sidebar
            st.image(logo, width=200)
        
        elif location == "header":
            # Opción 3: Logo junto al título
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(logo, width=100)
            with col2:
                st.title("Análisis de Producción Farmacéutica")
            return True
    return False


if not display_logo("header"):
    # Si no hay logo, mostrar título normal
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

CSV = "master_table_fixed3.csv"
PARQ = "master_volume.parquet"

df = pd.read_csv(CSV, sep=";", dtype="string")
df.to_parquet(PARQ, index=False)
print("✅ master_volume.parquet guardado:", Path(PARQ).stat().st_size/1e6, "MB")

def load_data():
    ART = Path("artifacts")
    
   # Lee el CSV y devuelve el DataFrame
 #   df = pd.read_csv("artifacts/master_table_fixed3.csv")
    df = pd.read_parquet("master_volume.parquet")
    with open(ART / "preprompt2.txt", 'r', encoding='utf-8') as f:
        preprompt = f.read()
    
    return df, preprompt
st.text(f"DEBUG: listo load data")
# askllm3 modificada para mantener contexto
def ask_llm3_with_context(question: str, conversation_history: list, years=None):
    client = init_openai_client()
    df, preprompt = load_data()
        
    # system_msg = preprompt + "\n\nDATOS (JSON por año):\n" + json.dumps(payload_multi, ensure_ascii=False, indent=2)
    # Armar el mensaje de sistema con preprompt + csv
    system_msg = (
        preprompt
        + "\n\nUsa estos datos tabulares en CSV:\n"
        + df.head(50).to_csv(index=False)  # Limitá la cantidad si no querés romper el límite de tokens
    )
    
    # Construir mensajes incluyendo el historial
    messages = [{"role": "system", "content": system_msg}]
    
    # Agregar el historial de conversación (limitado a los últimos 10 intercambios para no exceder límites)
    for msg in conversation_history[-20:]:  # Últimos 10 pares de pregunta-respuesta
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Agregar la pregunta actual
    messages.append({"role": "user", "content": question})
    
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=1000,
        temperature=0.3
    )
    
    return resp.choices[0].message.content.strip()

# History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agregar mensaje de bienvenida si es la primera vez
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

# Campo de entrada para el usuario
if prompt := st.chat_input("Haz una pregunta sobre los datos de producción..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando datos..."):
            try:
                # Usar la función modificada que incluye el contexto
                response = ask_llm3_with_context(prompt, st.session_state.messages)
                st.markdown(response)
                
                # Agregar respuesta al historial
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")

# Sidebar con información adicional
with st.sidebar:
    #Logo en el sidebar
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
    
    # Información sobre el contexto
    st.markdown(f"**💬 Mensajes en contexto:** {len(st.session_state.messages)}")
    
    # Botón para limpiar el historial
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun() 
