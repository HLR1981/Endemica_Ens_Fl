import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# --- ESTILOS VISUALES ---
st.markdown("""
<style>

/* Fondo con degradado */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #1a1d29 100%);
}

/* Título */
h1 {
    color: #FF4B8B;
    text-align: center;
    font-size: 32px;
}

/* Subtítulo */
.subtitle {
    text-align: center;
    color: #CCCCCC;
    font-size: 16px;
}

/* Texto general */
p, label {
    color: #EAEAEA;
}

/* Upload */
.stFileUploader {
    border: 2px dashed #FF4B8B;
    border-radius: 12px;
    padding: 10px;
}

/* Botones */
.stButton>button {
    background-color: #FF4B8B;
    color: white;
    border-radius: 12px;
    font-size: 16px;
}

/* Resultado */
.result-box {
    background-color: #1f3d2b;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

/* Expanders */
.streamlit-expanderHeader {
    font-size: 16px;
    color: #FF4B8B;
}

</style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.markdown("<h1>🌸 EndémicaEns</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Flora de Ensenada</p>", unsafe_allow_html=True)

# --- DICCIONARIO DE ESPECIES ---
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

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- SUBIDA DE ARCHIVO ---
archivo = st.file_uploader("📸 Sube una foto de la flora local", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo)
    st.image(img, use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])

    nombres_carpetas = sorted(list(especies_info.keys()))
    clase_detectada = nombres_carpetas[np.argmax(score)]
    confianza = 100 * np.max(score)

    info = especies_info[clase_detectada]

    # --- RESULTADO BONITO ---
    st.markdown(f"""
    <div class="result-box">
        <h3 style="color:#00FFAA;">{info['nombre']}</h3>
        <p><b>Confianza:</b> {confianza:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # --- DETALLES ---
    with st.expander("📖 Ver Detalles Técnicos"):
        st.write(f"**Científico:** *{info['cientifico']}*")
        st.write(f"**Estado:** {info['estado']}")
        st.info(info['info'])
        st.warning(f"**Cuidados:** {info['cuidados']}")
