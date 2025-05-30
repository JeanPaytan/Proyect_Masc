from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)

MODEL_PATH = 'modelo_mascarilla.h5'

# Si el modelo no está, descargarlo desde un enlace (ajustaremos esto más adelante)
if not os.path.exists(MODEL_PATH):
    url = 'https://drive.google.com/uc?export=download&id=1N55dDmlYJDKEyksu4MIhTIiQuWRB9dn4'# <-- esto lo actualizaremos
    r = requests.get(url)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

# Cargar modelo
model = load_model(MODEL_PATH)
labels = ['Sin mascarilla', 'Con mascarilla', 'Mal puesta']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió imagen'})

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]
    confidence = float(np.max(prediction[0]))

    return jsonify({'resultado': result, 'confianza': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
