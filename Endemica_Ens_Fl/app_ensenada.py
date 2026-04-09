import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO VISUAL ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# Inyectamos CSS para cambiar el fondo y los colores
st.markdown("""
    <style>
    /* Fondo principal color crema/arena */
    .stApp {
        background-color: #FDFBF0;
    }
    /* Estilo para los títulos */
    h1 {
        color: #2D4A22 !important;
        font-family: 'Helvetica', sans-serif;
    }
    /* Estilo para las tarjetas de información */
    .stAlert {
        border-radius: 15px;
    }
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
    # Asegúrate de que esta ruta sea correcta en tu GitHub
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

model = load_model()

# --- 3. INTERFAZ DE USUARIO ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto o captura desde tu cámara", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    img = Image.open(archivo)
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # Preprocesamiento
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Predicción
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    
    nombres_carpetas = sorted(list(especies_info.keys()))
    clase_detectada = nombres_carpetas[np.argmax(score)]
    confianza = 100 * np.max(score)

    # --- 4. LÓGICA DE IMAGEN DESCONOCIDA ---
    # Si la confianza es menor al 40%, diremos que no la reconoce
    if confianza < 40.0:
        st.error("### ⚠️ Imagen desconocida")
        st.write(f"La confianza es muy baja ({confianza:.2f}%).")
        st.info("Sugerencia: Intenta tomar la foto con mejor luz, más cerca de la flor o asegúrate de que sea una de las especies de nuestra guía.")
    else:
        # Si la confianza es alta, mostramos la información
        info = especies_info[clase_detectada]
        st.success(f"### Identificado como: {info['nombre']}")
        st.write(f"**Confianza:** {confianza:.2f}%")
        
        with st.expander("📖 Ver Detalles Técnicos y Cuidados"):
            st.write(f"**Científico:** *{info['cientifico']}*")
            st.write(f"**Estado:** {info['estado']}")
            st.divider()
            st.write("**Descripción:**")
            st.write(info['info'])
            st.warning(f"**💡 Cuidados:** {info['cuidados']}")
            st.info(f"**🗓️ Época de plantación:** {info['plantacion']}")
