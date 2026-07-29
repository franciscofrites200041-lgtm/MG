import re

def limpiar_texto(texto: str) -> str:
    """Repite la lógica del JS de n8n para eliminar caracteres que rompen Telegram y formato feo."""
    if not texto:
        return ""
    
    # Quitar escapes innecesarios
    texto = texto.replace("\\", "")
    
    # Quitar markdown fuerte de Gemini
    texto = texto.replace("**", "")
    texto = texto.replace("*", "")
    texto = texto.replace("__", "")
    texto = texto.replace("_", "")
    
    # Limpiar títulos markdown (# Titulo)
    texto = re.sub(r"^#+\s*", "", texto, flags=re.MULTILINE)
    
    # Limpiar enlaces MD dejándolos solo como texto
    texto = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", texto)
    
    return texto

def dividir_mensaje(texto: str, limite: int = 3800) -> list[str]:
    """Divide un string muy grande en varios mensajes para Telegram de manera inteligente por punto o espacio."""
    mensajes_salida = []
    
    if len(texto) > limite:
        inicio = 0
        while inicio < len(texto):
            fin = min(inicio + limite, len(texto))
            
            if fin < len(texto):
                ultimo_punto = texto.rfind('.', inicio, fin)
                ultimo_espacio = texto.rfind(' ', inicio, fin)
                
                # Intentar cortar post punto, sino en espacio, sino forzar corte.
                if ultimo_punto > inicio + 3000:
                    fin = ultimo_punto + 1
                elif ultimo_espacio > inicio + 3000:
                    fin = ultimo_espacio
                    
            corte = texto[inicio:fin].strip()
            if corte:
                mensajes_salida.append(corte)
            
            inicio = fin
    else:
        if texto.strip():
            mensajes_salida.append(texto.strip())
            
    return mensajes_salida
