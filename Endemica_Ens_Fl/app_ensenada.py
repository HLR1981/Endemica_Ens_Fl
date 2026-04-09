import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# CSS para que se vea "viva" la app
st.markdown("""
    <style>
    .stApp {
        background-color: #FDFBF0;
    }
    h1 {
        color: #2D4A22 !important;
    }
    .stAlert {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 EndémicaEns: Flora de Ensenada")
st.markdown("### Identificador de plantas nativas")

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
        "cuidados": "Evitar exceso de riego en el tronco durante el verano.",
        "plantacion": "Otoño."
    },
    "lila_california_ceanothus": {
        "nombre": "Lila de California",
        "cientifico": "Ceanothus spp.",
        "estado": "Nativa Regional",
        "info": "Famosa por sus racimos de flores azules o moradas. Atrae a muchas mariposas.",
        "cuidados": "Requiere poco riego una vez establecida.",
        "plantacion": "Otoño o Invierno."
    },
    "maguey de costa_agave_shawii": {
        "nombre": "Maguey de Costa",
        "cientifico": "Agave shawii",
        "estado": "Nativa Regional",
        "info": "Suculenta protegida que crece frente al mar. Sus flores atraen a polinizadores.",
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
        "info": "Arbusto aromático de flores moradas, esencial para abejas y colibríes.",
        "cuidados": "Sol directo y suelo bien drenado.",
        "plantacion": "Primavera."
    }
}

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # Asegúrate de que esta ruta sea la correcta en tu carpeta de GitHub
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- 4. INTERFAZ DE CARGA ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto o captura desde tu cámara", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    # Abrir y forzar formato RGB para evitar errores de canales (como en la palmera)
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # Preprocesamiento exacto para el modelo
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalización (0 a 1)
    img_array = tf.expand_dims(img_array, 0) # Crear el bloque para el modelo
    
    # Predicción
    with st.spinner('Identificando...'):
        pred = model.predict(img_array)
        # Usamos softmax si el modelo no tiene la capa integrada, 
        # o simplemente sacamos el índice más alto.
        score = tf.nn.softmax(pred[0])
        
        nombres_carpetas = sorted(list(especies_info.keys()))
        clase_detectada = nombres_carpetas[np.argmax(score)]
        confianza = 100 * np.max(score)

    # --- 5. RESULTADOS CON FILTRO DE CONFIANZA ---
    if confianza < 40.0:
        st.error("### ⚠️ Imagen desconocida")
        st.write(f"La confianza es muy baja ({confianza:.2f}%).")
        st.info("Sugerencia: Intenta tomar la foto más cerca de la planta o con mejor iluminación.")
    else:
        info = especies_info[clase_detectada]
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Nivel de confianza:** {confianza:.2f}%")
        
        with st.expander("📖 Ver Detalles Técnicos y Cuidados"):
            st.write(f"**Nombre Científico:** *{info['cientifico']}*")
            st.write(f"**Estado en la región:** {info['estado']}")
            st.divider()
            st.write(f"**Descripción:** {info['info']}")
            st.warning(f"**💡 Cuidados:** {info['cuidados']}")
            st.info(f"**🗓️ Época ideal de plantación:** {info['plantacion']}")
