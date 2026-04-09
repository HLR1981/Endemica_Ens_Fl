import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO ---
st.set_page_config(page_title="EndémicaEns", page_icon="🌸")

# Estilo para que se vea más profesional y "viva"
st.markdown("""
    <style>
    .stApp {
        background-color: #FDFBF0;
    }
    h1 {
        color: #2D4A22 !important;
        font-family: 'Helvetica', sans-serif;
    }
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
    # Ruta al archivo .keras dentro de tu repositorio
    return tf.keras.models.load_model('Endemica_Ens_Fl/modelo_flora_ensenada.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- 4. INTERFAZ DE CARGA ---
st.write("---")
archivo = st.file_uploader("🌿 Sube una foto o captura desde tu cámara", type=["jpg", "png", "jpeg", "webp"])

if archivo:
    # PASO A: Abrir y convertir a RGB (Esto evita el ValueError con transparencias)
    img = Image.open(archivo).convert("RGB")
    st.image(img, width=400, caption="Imagen seleccionada")
    
    # PASO B: Preprocesamiento exacto
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    
    # PASO C: Normalización (CRÍTICO para evitar el error de la palmera/salvia)
    img_array = img_array / 255.0 
    
    # PASO D: Ajustar forma a (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- 5. PREDICCIÓN ---
    with st.spinner('Analizando planta...'):
        try:
            pred = model.predict(img_array)
            # Aplicamos softmax para obtener probabilidades claras
            score = tf.nn.softmax(pred[0])
            
            # Obtener la clase con mayor probabilidad
            nombres_carpetas = sorted(list(especies_info.keys()))
            indice_maximo = np.argmax(score)
            clase_detectada = nombres_carpetas[indice_maximo]
            confianza = 100 * np.max(score)

            # --- 6. RESULTADOS ---
            # Si la confianza es menor a 35%, consideramos que no la conoce
            if confianza < 35.0:
                st.error("### ⚠️ Imagen desconocida")
                st.write(f"La IA no está segura ({confianza:.2f}%).")
                st.info("Intenta que la foto esté más enfocada o usa una de las 6 plantas de la guía.")
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
                    st.info(f"**🗓️ Época de plantación:** {info['plantacion']}")
        
        except Exception as e:
            st.error(f"Error técnico durante la predicción: {e}")
