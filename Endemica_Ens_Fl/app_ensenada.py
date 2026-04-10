import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# --- ESTILOS VISUALES ---
st.markdown("""
<style>

/* Fondo */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #1a1d29 100%);
}

/* Título */
h1 {
    color: #FF4B8B;
    text-align: center;
    font-size: 34px;
    font-weight: bold;
}

/* Subtítulo */
.subtitle {
    text-align: center;
    color: #FFFFFF;
    font-size: 18px;
    margin-bottom: 20px;
}

/* Labels */
label, .stMarkdown, .stText {
    color: #FFFFFF !important;
    font-size: 16px;
}

/* Texto */
p {
    color: #EAEAEA;
}

/* Upload */
.stFileUploader {
    border: 2px dashed #FF4B8B;
    border-radius: 12px;
    padding: 10px;
}

/* Resultado */
.result-box {
    background-color: #1f3d2b;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    color: white;
}

/* Expanders */
.streamlit-expanderHeader {
    color: #FF4B8B !important;
    font-weight: bold;
}

/* Mensajes */
.stSuccess {
    color: #00FFAA;
}

.stError {
    color: #FF6B6B;
}

</style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.markdown("<h1>🌸 EndémicaEns</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Flora de Ensenada</p>", unsafe_allow_html=True)

# --- DICCIONARIO ---
especies_info = {
    "encelia farinosa": {
        "nombre": "Incienso / Encelia",
        "cientifico": "Encelia farinosa",
        "estado": "Nativa",
        "info": "Arbusto de flores amarillas.",
        "cuidados": "Muy poca agua, pleno sol.",
        "plantacion": "Primavera."
    },
    "encino_quercus_agrifolia": {
        "nombre": "Encino Californiano",
        "cientifico": "Quercus agrifolia",
        "estado": "Nativa / Protegida",
        "info": "Árbol fundamental para el ecosistema.",
        "cuidados": "Evitar exceso de riego.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Flores azules o moradas.",
        "cuidados": "Poco riego.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de costa_agave_shawii": {
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida.",
        "cuidados": "Suelo arenoso.",
        "plantacion": "Invierno."
    },
    "rosa de castlla_rosa_minutifolia": {
        "nombre": "Rosa de Castilla",
        "cientifico": "Rosa minutifolia",
        "estado": "Endémica de BC",
        "info": "Flor rosa vibrante.",
        "cuidados": "Sin riego.",
        "plantacion": "Lluvias."
    },
    "salvia de munz_salvia_munzii": {
        "nombre": "Salvia de Munz",
        "cientifico": "Salvia munzii",
        "estado": "Nativa Regional",
        "info": "Flores moradas.",
        "cuidados": "Sol directo.",
        "plantacion": "Primavera."
    }
}

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- UPLOAD ---
archivo = st.file_uploader("📸 Sube una foto de la flora local", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    try:
        # Convertir imagen
        img = Image.open(archivo).convert("RGB")
        st.image(img, use_column_width=True)

        # 🔥 TAMAÑO CORRECTO (160x160)
        img_resized = img.resize((160, 160))

        # Convertir a array
        img_array = np.array(img_resized).astype("float32")

        # 🔥 Preprocesamiento MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Expandir dimensiones
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
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
            st.write(f"**Científico:** {info['cientifico']}")
            st.write(f"**Estado:** {info['estado']}")
            st.info(info['info'])
            st.warning(f"Cuidados: {info['cuidados']}")

    except Exception as e:
        st.error("⚠️ Error procesando la imagen o el modelo")
        st.text(str(e))
