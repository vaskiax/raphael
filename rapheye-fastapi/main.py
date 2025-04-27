import logging
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Carga diferida del modelo Pix2Tex ---
ocr_model = None
PIX2TEX_AVAILABLE = False

def load_pix2tex_model():
    global ocr_model, PIX2TEX_AVAILABLE
    if ocr_model is None:
        logger.info("Intentando cargar pix2tex y el modelo OCR...")
        try:
            # Importar aquí para que el resto de la app pueda iniciar
            # incluso si pix2tex tiene problemas al importar
            from pix2tex.cli import LatexOCR
            ocr_model = LatexOCR()
            PIX2TEX_AVAILABLE = True
            logger.info("pix2tex y modelo OCR cargados exitosamente.")
        except ImportError as e:
            PIX2TEX_AVAILABLE = False
            logger.error(f"Error CRÍTICO al importar pix2tex: {e}. La función OCR no estará disponible.")
        except Exception as e:
            PIX2TEX_AVAILABLE = False
            logger.error(f"Error CRÍTICO inesperado al cargar pix2tex/modelo: {e}.")
    return PIX2TEX_AVAILABLE

# --- Inicialización de la App FastAPI ---
app = FastAPI(title="Rapheye API", description="API para extraer LaTeX de imágenes de ecuaciones usando Pix2Tex.")

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Intenta cargar el modelo al iniciar la app para el "arranque en caliente"
    # Si falla, la app seguirá corriendo pero /predict dará error.
    load_pix2tex_model()

@app.get("/", summary="Endpoint de Salud", tags=["General"])
async def root():
    """Verifica si la API está en ejecución."""
    model_status = "disponible" if PIX2TEX_AVAILABLE and ocr_model else "NO disponible"
    return {"message": f"Rapheye API funcionando. Estado del modelo OCR: {model_status}"}

@app.post("/predict/", summary="Extraer LaTeX de Imagen", tags=["OCR"])
async def predict_equation(file: UploadFile = File(..., description="Archivo de imagen con la ecuación.")):
    """
    Recibe una imagen, usa Pix2Tex para extraer la ecuación en formato LaTeX.
    """
    if not PIX2TEX_AVAILABLE or ocr_model is None:
        logger.error("Intento de predicción fallido: modelo OCR no cargado.")
        raise HTTPException(status_code=503, detail="Servicio OCR no disponible en este momento.")

    logger.info(f"Recibida imagen: {file.filename}, tipo: {file.content_type}")
    try:
        # Leer contenido del archivo subido
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info("Imagen procesada con Pillow.")

        # Realizar predicción (puede tardar)
        logger.info("Realizando predicción LaTeX con pix2tex...")
        latex_result = ocr_model(image)
        logger.info("Predicción Pix2Tex completada.")

        if latex_result:
            logger.info(f"LaTeX extraído: {latex_result[:100]}...")
            return {"latex": latex_result}
        else:
            logger.warning("Pix2Tex no devolvió resultado para la imagen.")
            raise HTTPException(status_code=404, detail="No se pudo extraer LaTeX de la imagen.")

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}", exc_info=True) # Log con traceback
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar la imagen: {e}")
    finally:
         await file.close()

# --- (Opcional) Configurar uvicorn si se ejecuta directamente ---
# Esto no es necesario para App Service, que usará el comando de inicio
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)