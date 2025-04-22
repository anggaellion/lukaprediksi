from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/uploaded'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load model
model = tf.keras.models.load_model('model/model_prediksi_lukaluar.h5')

# Fungi preprocessing gambar
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    try:
        processed_image = preprocess_image(save_path)
        prediction = model.predict(processed_image)
        
        # Sesuaikan dengan output model Anda
        class_names = ['abrasi', 'bakar', 'laserasi', 'lebam', 'sayat', 'tusuk']  # Contoh kelas
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)