import streamlit as st
import sounddevice as sd
import wavio
from datetime import datetime
import os # Necesario para la limpieza del archivo

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Grabador Simple de Audio", layout="centered")
st.title("🎤 Grabador de Audio Sencillo")
st.write("Haz clic en el botón, graba 10 segundos de audio, y se descargará automáticamente.")

# --- Parámetros Fijos de Grabación ---
DURATION = 10  # segundos
SAMPLERATE = 44100 # Hz (calidad de CD)
CHANNELS = 1 # Mono

# --- Función para Grabar Audio ---
def record_and_save_audio(duration, samplerate, channels):
    filename = f"grabacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    st.info(f"Grabando audio por {duration} segundos... ¡Habla ahora!")
    
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
        sd.wait() # Espera a que la grabación termine
        
        wavio.write(filename, recording, samplerate, sampwidth=2)
        st.success(f"Grabación terminada. Archivo guardado como: '{filename}'")
        return filename
    except Exception as e:
        st.error(f"Error al grabar: {e}. Asegúrate de tener un micrófono conectado y permisos.")
        return None

# --- Interfaz de Usuario de Streamlit ---
if st.button("🔴 Grabar 10 Segundos y Descargar"):
    recorded_file_path = record_and_save_audio(DURATION, SAMPLERATE, CHANNELS)
    
    if recorded_file_path:
        # Se muestra un reproductor de audio después de grabar
        st.subheader("Tu Grabación:")
        st.audio(recorded_file_path)

        # Ofrecer la descarga automática
        with open(recorded_file_path, "rb") as file:
            st.download_button(
                label="Haz clic para descargar el archivo WAV",
                data=file.read(),
                file_name=os.path.basename(recorded_file_path),
                mime="audio/wav"
            )
        
        # Opcional: Eliminar el archivo después de que se ha ofrecido para descargar
        # Esto es útil para mantener el sistema limpio, especialmente en Streamlit Cloud
        os.remove(recorded_file_path)
        st.info("Archivo local eliminado para limpiar el espacio.")