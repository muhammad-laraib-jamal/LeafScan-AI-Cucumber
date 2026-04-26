from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model/model.h5')

# Class names
classes = ['Healthy', 'Powdery Mildew', 'Leaf Spot']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # ✅ Generate unique filename
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # ✅ Save file ONCE
    file.save(filepath)

    # Process image
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    result = classes[np.argmax(prediction)]

    return render_template(
        'index.html',
        prediction=result,
        confidence=round(confidence, 2),
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)