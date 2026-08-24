# Functions de Open WebUI

Scripts Python que se instalan en el Admin Panel de Open WebUI para agregar features custom.

## Instalar `descargar_docx.py`

1. En https://bot-estudio.montoya-gherzi.com.ar/ login como **admin**
2. **Panel de Administración** → **Functions** → **+ Nueva función**
3. Copiar el contenido completo de `descargar_docx.py` y pegar en el editor
4. **Guardar** (Open WebUI instala automaticamente `python-docx` si falta)
5. Activar el toggle **Enabled**
6. En **Models** → editar el modelo `mg-bot` → **Functions** → tildar "Descargar como DOCX"

## Uso

En cada respuesta del bot aparece un botón 📄 debajo del mensaje. Click y aparece el link "**Descargar escrito_montoya_gherzi.docx**". Click descarga el archivo con formato legal (Times New Roman 12, A4, márgenes 2.5cm/3cm).

## Formato aplicado

- Fuente: Times New Roman 12pt
- Márgenes: 2.5cm sup/inf, 3cm izq, 2cm der (procesal argentino)
- Headings markdown (`#`, `##`, `###`) → Heading 1/2/3
- Bullets `-`/`*` → List Bullet
- Numeradas `1.` → List Number
- **Negrita** markdown → run.bold
- Lineas en MAYUSCULAS → centrado + negrita (encabezados juridicos)

## Config (Valves)

Cada function tiene Valves editables desde la UI de admin:
- `font_name` (default: Times New Roman)
- `font_size_pt` (default: 12)
- `margin_*_cm` (default: 2.5/2.5/3/2)
