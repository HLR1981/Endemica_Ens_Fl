import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# Estilo visual
st.markdown("""
    <style>
    .stApp { background-color: #FDFBF0; }
    h1 { color: #2D4A22 !important; font-family: 'Helvetica', sans-serif; }
    .stAlert { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 EndémicaEns: Flora de Ensenada")
st.markdown("### Identificador inteligente de plantas nativas")

# --- 2. DICCIONARIO DE INFORMACIÓN ---
especies_info = {
    "encelia farinosa": {"nombre": "Incienso / Encelia", "info": "Arbusto de flores amarillas."},
    "encino_quercus_agrifolia": {"nombre": "Encino Californiano", "info": "Árbol majestuoso regional."},
    "lila_california_ceanothus": {"nombre": "Lila de California", "info": "Flores azules o moradas."},
    "maguey de costa_agave_shawii": {"nombre": "Maguey de Costa", "info": "Suculenta protegida frente al mar."},
    "rosa de castlla_rosa_minutifolia": {"nombre": "Rosa de Castilla", "info": "Joya endémica de Ensenada."},
    "salvia de munz_salvia_munzii": {"nombre": "Salvia de Munz", "info": "Arbusto aromático de flores moradas."}
}

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # Asegúrate de que esta ruta sea la correcta en tu GitHub
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- 4. INTERFAZ DE CARGA ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto de la planta", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    # Preprocesamiento: Convertir a RGB y redimensionar a 160x160
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    img_resized = img.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized) / 255.0 # Normalización
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- 5. PREDICCIÓN ---
    with st.spinner('Identificando...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
        
        # Este orden es el que vimos en tu carpeta 'train'
        nombres_lista = [
            "encelia farinosa",              # 0
            "encino_quercus_agrifolia",      # 1
            "lila_california_ceanothus",     # 2
            "maguey de costa_agave_shawii",  # 3
            "rosa de castlla_rosa_minutifolia", # 4
            "salvia de munz_salvia_munzii"   # 5
        ]
        
        idx = np.argmax(score)
        confianza = 100 * np.max(score)

    # --- LÍNEA DE TRUCO (DEBUG) ---
    # Esto nos dirá qué número asigna el modelo a cada foto
    st.warning(f"DEBUG: El modelo detectó el Índice: {idx}")

    # --- 6. RESULTADOS ---
    if confianza < 45.0: # Umbral de seguridad
        st.error(f"### ⚠️ Imagen desconocida ({confianza:.2f}%)")
        st.info("La confianza es baja. Intenta acercar más la cámara a la planta.")
    else:
        # Buscamos la información usando el índice detectado
        clase_detectada = nombres_lista[idx]
        info = especies_info[clase_detectada]
        
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Confianza:** {confianza:.2f}%")
        st.info(f"**Información:** {info['info']}")
