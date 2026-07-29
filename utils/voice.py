from openai import AsyncOpenAI
import os
import aiofiles

# Inicializar cliente asíncrono
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def transcribir_audio(ruta_archivo_local: str) -> str:
    """Envía un archivo de audio local a la API de OpenAI Whisper para su transcripción y devuelve el texto."""
    if not os.path.exists(ruta_archivo_local):
        return ""
        
    try:
        # Abrir el archivo de manera tradicional en modo lectura binaria ya que openai exige el objeto de archivo
        with open(ruta_archivo_local, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="es" # Definido explícitamente en el json
            )
            return transcript.text
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error en transcripción de audio: {e}")
        return ""
    finally:
        # Intentar borrar el archivo local para no ocupar espacio en el NAS
        try:
            if os.path.exists(ruta_archivo_local):
                os.remove(ruta_archivo_local)
        except:
            pass
