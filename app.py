from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = load_model('models/inception_best_model.keras')  # Load your trained model

@app.route('/info.html')
def info():
    return render_template('info.html')

@app.route('/aboutus.html')
def aboutus():
    return render_template('aboutus.html')

@app.route('/treatment.html')
def treatment():
    return render_template('treatment.html')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path, lower_threshold=0.4, upper_threshold=0.6):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image

    prediction = model.predict(img_array)
    confidence = prediction[0][0]  # Get the confidence score

    print(f"Prediction Confidence: {confidence}")  # Print the raw confidence score for debugging

    # Check confidence thresholds
    if confidence > upper_threshold:
        return "Oscc"
    elif confidence < lower_threshold:
        return "Normal"
    else:
        return "Please provide a correct input image."
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('index.html', prediction=prediction, image_path=filepath)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
