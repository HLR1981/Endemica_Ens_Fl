import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

st.markdown("""
    <style>
    .stApp { background-color: #FDFBF0; }
    h1 { color: #2D4A22 !important; font-family: 'Helvetica', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 EndémicaEns: Flora de Ensenada")

# --- 2. DICCIONARIO ---
especies_info = {
    "encelia farinosa": {"nombre": "Incienso / Encelia", "info": "Arbusto de flores amarillas."},
    "encino_quercus_agrifolia": {"nombre": "Encino Californiano", "info": "Árbol majestuoso regional."},
    "lila_california_ceanothus": {"nombre": "Lila de California", "info": "Flores azules o moradas."},
    "maguey de costa_agave_shawii": {"nombre": "Maguey de Costa", "info": "Suculenta protegida que crece cerca del mar."},
    "rosa de castlla_rosa_minutifolia": {"nombre": "Rosa de Castilla", "info": "Planta endémica única de Baja California."},
    "salvia de munz_salvia_munzii": {"nombre": "Salvia de Munz", "info": "Arbusto aromático de flores moradas."}
}

# --- 3. MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- 4. INTERFAZ ---
archivo = st.file_uploader("🌿 Sube una foto", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400)
    
    # Procesamiento (160x160)
    img_array = np.array(img.resize((160, 160))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- 5. PREDICCIÓN ---
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    
    # ESTE ORDEN DEBERÍA SER EL DEFINITIVO:
    nombres_lista = [
        "encelia farinosa",              # 0
        "encino_quercus_agrifolia",      # 1
        "lila_california_ceanothus",     # 2
        "salvia de munz_salvia_munzii",   # 3
        "rosa de castlla_rosa_minutifolia", # 4
        "maguey de costa_agave_shawii"    # 5
    ]
    
    idx = np.argmax(score)
    confianza = 100 * np.max(score)

    # --- 6. RESULTADOS ---
    if confianza < 45.0:
        st.error(f"⚠️ Imagen desconocida ({confianza:.2f}%)")
    else:
        info = especies_info[nombres_lista[idx]]
        st.success(f"### {info['nombre']}")
        st.write(f"**Confianza:** {confianza:.2f}%")
        st.info(info['info'])
