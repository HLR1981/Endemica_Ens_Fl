import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")
st.title("🌸 EndémicaEns: Flora de Ensenada")

# --- DICCIONARIO ACTUALIZADO CON 6 ESPECIES ---
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
        "info": "Árbol majestuoso de los arroyos de Ensenada. Es fundamental para el ecosistema local.",
        "cuidados": "Evitar exceso de riego en el tronco durante el verano.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Famosa por sus racimos de flores azules o moradas. Es muy resistente y atrae a muchas mariposas.",
        "cuidados": "Requiere poco riego una vez establecida y suelo bien drenado.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de costa_agave_shawii": {
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida que crece frente al mar. Sus flores atraen a murciélagos y aves.",
        "cuidados": "Suelo arenoso y mucha brisa marina.",
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
        "info": "Arbusto aromático de flores moradas, esencial para los polinizadores de la zona.",
        "cuidados": "Sol directo y suelo bien drenado.",
        "plantacion": "Primavera."
    }
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_flora_ensenada.keras')

model = load_model()

archivo = st.file_uploader("Sube una foto de la flora local", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo)
    st.image(img, width=400)
    
    img_resized = img.resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    
    nombres_carpetas = sorted(list(especies_info.keys()))
    clase_detectada = nombres_carpetas[np.argmax(score)]
    confianza = 100 * np.max(score)

    info = especies_info[clase_detectada]
    st.success(f"### {info['nombre']}")
    st.write(f"**Confianza:** {confianza:.2f}%")
    
    with st.expander("📖 Ver Detalles Técnicos"):
        st.write(f"**Científico:** *{info['cientifico']}*")
        st.write(f"**Estado:** {info['estado']}")
        st.info(info['info'])
        st.warning(f"**Cuidados:** {info['cuidados']}")