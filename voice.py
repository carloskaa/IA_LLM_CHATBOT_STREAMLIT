import streamlit as st
import whisper
import os
import tempfile
import subprocess
import base64

# ---------------- CONFIGURACIÓN ----------------
cohere_api_key = '1msKL9N3DxmNqmxMCQLQ4CHz8e1dO130v1urBoUI'
st.set_page_config(page_title="Chatbot Científico", layout="centered")

# ---------------- RUTA MANUAL A FFMPEG ----------------
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # ✅ AJUSTA esta ruta a donde tengas ffmpeg.exe

if not os.path.isfile(ffmpeg_path):
    st.error(f"No se encontró ffmpeg.exe en: {ffmpeg_path}")
else:
    # Parcheo para que Whisper use la ruta personalizada de ffmpeg
    def patched_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            cmd[0] = ffmpeg_path
        return subprocess.run(cmd, **kwargs)
    whisper.audio.run = patched_run  # 👈 se reemplaza la función por la parcheada

# ---------------- FONDO ----------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            background-color: transparent !important;
            color: white;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: #FFFFFF !important;
        }}
        .stTextInput > div > div > input {{
            background-color: #1b1f2a;
            color: white;
        }}
        div.stButton > button:first-child {{
            background-color: black !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #555;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("fondo.jpeg")

# ---------------- TÍTULO E IMAGEN ----------------
st.markdown("<h1 style='text-align: center; color: #A1F55A;'>Chatbot <span style='color:#8BF'>SCIENTIA</span></h1>", unsafe_allow_html=True)

with open("logo2.jpeg", "rb") as image_file:
    logo2_encoded = base64.b64encode(image_file.read()).decode()
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{logo2_encoded}' width='100'/></div>", unsafe_allow_html=True)

st.markdown("<h3 style='color: white; text-align: center;'>Bienvenido, puedes escribir tu pregunta o subir una nota de voz.</h3>", unsafe_allow_html=True)

# ---------------- INPUT DE AUDIO ----------------
st.markdown("### Subir una nota de voz (.m4a, .mp3, .wav)")
audio_file = st.file_uploader("Selecciona un archivo de audio", type=["m4a", "mp3", "wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Transcribiendo con Whisper..."):
        try:
            model = whisper.load_model("base")  # Puedes usar "small" o "medium" si tienes GPU
            result = model.transcribe(tmp_path, language="Spanish")
            transcription = result["text"]
            st.success("✅ Transcripción:")
            st.markdown(f"<p style='color: white;'>{transcription}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error durante la transcripción: {e}")

# ---------------- OPCIONAL: CAMPO DE TEXTO ----------------
user_text = st.text_input("¿O prefieres escribir tu pregunta?", key="input_text")

if st.button("Enviar texto"):
    if user_text:
        st.success(f"Tú escribiste: {user_text}")
    else:
        st.warning("Escribe algo para enviar.")
