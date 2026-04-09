import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO ---
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

# --- 2. DICCIONARIO DE ESPECIES ---
especies_info = {
    "encelia farinosa": {
        "nombre": "Incienso / Encelia",
        "cientifico": "Encelia farinosa",
        "estado": "Nativa",
        "info": "Arbusto de flores amarillas. Sus hojas grisáceas reflejan la luz para sobrevivir al calor.",
        "cuidados": "Muy poca agua, pleno sol.",
        "plantacion": "Primavera."
    },
    "encino_quercus_agrifolia": {
        "nombre": "Encino Californiano",
        "cientifico": "Quercus agrifolia",
        "estado": "Nativa / Protegida",
        "info": "Árbol majestuoso de los arroyos de Ensenada. Fundamental para el ecosistema local.",
        "cuidados": "Evitar exceso de riego en el tronco.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Famosa por sus racimos de flores azules o moradas. Atrae mariposas.",
        "cuidados": "Requiere poco riego una vez establecida.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de costa_agave_shawii": {
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida que crece frente al mar. Atrae polinizadores.",
        "cuidados": "Suelo arenoso y brisa marina.",
        "plantacion": "Invierno."
    },
    "rosa de castlla_rosa_minutifolia": {
        "nombre": "Rosa de Castilla",
        "cientifico": "Rosa minutifolia",
        "estado": "Endémica de BC",
        "info": "La joya de Ensenada. Pequeña, espinosa y con flores rosas vibrantes.",
        "cuidados": "Cero riego una vez establecida.",
        "plantacion": "Temporada de lluvias."
    },
    "salvia de munz_salvia_munzii": {
        "nombre": "Salvia de Munz",
        "cientifico": "Salvia munzii",
        "estado": "Nativa Regional",
        "info": "Arbusto aromático de flores moradas, esencial para abejas y colibríes.",
        "cuidados": "Sol directo y suelo bien drenado.",
        "plantacion": "Primavera."
    }
}

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- 4. INTERFAZ DE CARGA ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto de la planta", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # Preprocesamiento a 160x160 (el tamaño que tu modelo pide)
    img_resized = img.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- 5. PREDICCIÓN ---
    with st.spinner('Analizando...'):
        pred = model.predict(img_array)
        score = tf.nn.softmax(pred[0])
        
        # ORDEN MANUAL: Si la Rosa sale como Salvia, cambia el orden aquí abajo.
        nombres_lista = [
            "encelia farinosa", 
            "encino_quercus_agrifolia", 
            "lila_california_ceanothus", 
            "maguey de costa_agave_shawii", 
            "rosa de castlla_rosa_minutifolia", 
            "salvia de munz_salvia_munzii"
        ]
        
        indice_detectado = np.argmax(score)
        clase_detectada = nombres_lista[indice_detectado]
        confianza = 100 * np.max(score)

    # --- 6. RESULTADOS ---
    if confianza < 65.0:
        st.error(f"### ⚠️ Imagen desconocida ({confianza:.2f}%)")
        st.info("La IA no está segura. Prueba acercándote más a la flor.")
    else:
        info = especies_info[clase_detectada]
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Nivel de confianza:** {confianza:.2f}%")
        
        with st.expander("📖 Ver Detalles Técnicos y Cuidados"):
            st.write(f"**Nombre Científico:** *{info['cientifico']}*")
            st.write(f"**Estado:** {info['estado']}")
            st.divider()
            st.write(f"**Descripción:** {info['info']}")
            st.warning(f"**💡 Cuidados:** {info['cuidados']}")
