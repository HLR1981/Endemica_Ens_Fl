import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

st.markdown("""
    <style>
    .stApp { background-color: #FDFBF0; }
    h1 { color: #2D4A22 !important; font-family: 'Helvetica', sans-serif; }
    .stAlert { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 EndémicaEns: Flora de Ensenada")
st.markdown("### Identificador inteligente de plantas nativas")

# --- DICCIONARIO DE INFORMACIÓN ---
especies_info = {
    "encelia farinosa": {"nombre": "Incienso / Encelia", "info": "Arbusto de flores amarillas nativo de zonas áridas."},
    "encino_quercus_agrifolia": {"nombre": "Encino Californiano", "info": "Árbol majestuoso y protegido de Ensenada."},
    "lila_california_ceanothus": {"nombre": "Lila de California", "info": "Arbusto con hermosas flores azules o moradas."},
    "maguey de costa_agave_shawii": {"nombre": "Maguey de Costa", "info": "Suculenta protegida que crece cerca del mar."},
    "rosa de castlla_rosa_minutifolia": {"nombre": "Rosa de Castilla", "info": "Planta endémica única de Baja California."},
    "salvia de munz_salvia_munzii": {"nombre": "Salvia de Munz", "info": "Arbusto aromático de flores moradas."}
}

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- INTERFAZ ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto de la planta", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # Procesamiento (160x160 y normalización)
    img_resized = img.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('Identificando...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
        
        # EL ORDEN QUE FUNCIONÓ:
        nombres_lista = [
            "encelia farinosa", 
            "encino_quercus_agrifolia", 
            "lila_california_ceanothus", 
            "salvia de munz_salvia_munzii",     # Salvia
            "maguey de costa_agave_shawii",    # Maguey
            "rosa de castlla_rosa_minutifolia"  # Rosa al final
        ]
        
        idx = np.argmax(score)
        confianza = 100 * np.max(score)

    # --- RESULTADOS ---
    if confianza < 45.0:
        st.error(f"### ⚠️ Imagen desconocida ({confianza:.2f}%)")
        st.info("La IA no está segura. Prueba acercándote más a la planta.")
    else:
        clave = nombres_lista[idx]
        info = especies_info[clave]
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Nivel de confianza:** {confianza:.2f}%")
        st.info(f"**Información:** {info['info']}")
