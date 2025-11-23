import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image

# ==================================
# KONFIGURASI MODEL
# ==================================
MODEL_PATH = 'fruit_classifier.h5'
IMAGE_SIZE = 224

CLASS_LABELS = [
    "apple fruit", "banana fruit", "cherry fruit", "chickoo fruit",
    "grapes fruit", "kiwi fruit", "mango fruit", "orange fruit",
    "strawberry fruit"
]

model = None
app = FastAPI()

# ==================================
# KONFIGURASI CORS
# ==================================

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================================
# FUNGSI PRA-PEMROSESAN
# ==================================

def load_and_preprocess_image(file_content: bytes, target_size: int) -> np.ndarray:
    """Memuat dan memproses gambar."""
    try:
        img = Image.open(io.BytesIO(file_content))
        img = img.resize((target_size, target_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {e}")

# ==================================
# LIFECYCLE HOOKS
# ==================================

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model {MODEL_PATH} berhasil dimuat.")
    except Exception as e:
        print(f"GAGAL memuat model: {e}")
        raise RuntimeError("Model gagal dimuat.")

# ==================================
# ENDPOINT PREDIKSI
# ==================================

@app.post("/predict/fruit")
async def predict_fruit(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    file_content = await file.read()
    processed_image = load_and_preprocess_image(file_content, IMAGE_SIZE)

    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    predicted_label = CLASS_LABELS[predicted_index] if predicted_index < len(CLASS_LABELS) else "Kelas tidak dikenal"

    return {
        "filename": file.filename,
        "prediction": predicted_label,
        "confidence": confidence,
        "all_predictions": {
            CLASS_LABELS[i]: float(predictions[0][i])
            for i in range(len(CLASS_LABELS))
        }
    }

# ==================================
# ENDPOINT ROOT
# ==================================

@app.get("/")
def read_root():
    return {
        "status": "Model Fruit Classifier Siap Digunakan",
        "model_loaded": model is not None
    }
