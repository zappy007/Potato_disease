from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

app = FastAPI(title='AgroAI Production Engine')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

MODEL = tf.keras.layers.TFSMLayer('saved_model_dir', call_endpoint='serving_default')

@app.get('/')
def root():
    return {'status': 'active', 'engine': 'Render Free AI Core'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    bytes_data = await file.read()
    image = Image.open(BytesIO(bytes_data)).convert('RGB').resize((256, 256))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch  = np.expand_dims(img_array, axis=0)
    output      = MODEL(img_batch)
    predictions = list(output.values())[0].numpy()[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence      = float(np.max(predictions))
    return {'class': predicted_class, 'confidence': confidence}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10000)
