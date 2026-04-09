import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO ---
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

# --- 2. DICCIONARIO DE INFORMACIÓN (Basado en tus carpetas) ---
especies_info = {
    "encelia farinosa": {
        "nombre": "Incienso / Encelia",
        "cientifico": "Encelia farinosa",
        "info": "Arbusto de flores amarillas. Sus hojas grisáceas reflejan la luz para sobrevivir al calor.",
        "cuidados": "Muy poca agua, pleno sol."
    },
    "encino_quercus_agrifolia": {
        "nombre": "Encino Californiano",
        "cientifico": "Quercus agrifolia",
        "info": "Árbol majestuoso de los arroyos de Ensenada. Es fundamental para el ecosistema local.",
        "cuidados": "Evitar exceso de riego en el tronco."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "info": "Famosa por sus racimos de flores azules o moradas. Atrae a muchas mariposas.",
        "cuidados": "Requiere poco riego una vez establecida."
    },
    "maguey de costa_agave_shawii": {
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "info": "Suculenta protegida que crece frente al mar. Sus flores atraen a polinizadores.",
        "cuidados": "Suelo arenoso y mucha brisa marina."
    },
    "rosa de castlla_rosa_minutifolia": {
        "nombre": "Rosa de Castilla",
        "cientifico": "Rosa minutifolia",
        "info": "La joya de Ensenada. Pequeña, espinosa y con flores rosas vibrantes.",
        "cuidados": "Cero riego una vez establecida (planta de lluvia)."
    },
    "salvia de munz_salvia_munzii": {
        "nombre": "Salvia de Munz",
        "cientifico": "Salvia munzii",
        "info": "Arbusto aromático de flores moradas, esencial para abejas y colibríes.",
        "cuidados": "Sol directo y suelo bien drenado."
    }
}

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- 4. INTERFAZ DE USUARIO ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto de la planta", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    # Procesamiento de imagen
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # Redimensionar a 160x160 y normalizar
    img_resized = img.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- 5. PREDICCIÓN ---
    with st.spinner('Identificando especie...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
        
        # LISTA MANUAL SEGÚN EL ORDEN DE TUS CARPETAS EN WINDOWS
        nombres_lista = [
            "encelia farinosa",              # Índice 0
            "encino_quercus_agrifolia",      # Índice 1
            "lila_california_ceanothus",     # Índice 2
            "maguey de costa_agave_shawii",  # Índice 3
            "rosa de castlla_rosa_minutifolia", # Índice 4
            "salvia de munz_salvia_munzii"   # Índice 5
        ]
        
        indice_detectado = np.argmax(score)
        clase_detectada = nombres_lista[indice_detectado]
        confianza = 100 * np.max(score)

    # --- 6. RESULTADOS ---
    if confianza < 48.0:
        st.error(f"### ⚠️ Imagen desconocida ({confianza:.2f}%)")
        st.info("La IA no está segura. Intenta tomar la foto más cerca de la flor.")
    else:
        info = especies_info[clase_detectada]
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Nivel de confianza:** {confianza:.2f}%")
        
        with st.expander("📖 Ver Detalles Técnicos y Cuidados"):
            st.write(f"**Científico:** *{info['cientifico']}*")
            st.divider()
            st.write(f"**Descripción:** {info['info']}")
            st.warning(f"**💡 Cuidados:** {info['cuidados']}")
