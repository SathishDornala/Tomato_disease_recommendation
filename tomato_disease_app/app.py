from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import sys
import io

# Set system-wide UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the trained model
model = load_model(r'C:\Users\Sathish\tomato_disease_app\model\tomato_disease_recommendations.h5')

# Define the class names and pesticide recommendations
# Verify that the indices here match with the model's output indices
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Leaf_Mold', 'Tomato___Late_blight', 'Tomato___healthy']
pesticide_recommendations = {
    'Tomato___Bacterial_spot': 'Use Copper-based fungicides',
    'Tomato___Early_blight': 'Use Chlorothalonil or Azoxystrobin',
    'Tomato___healthy': 'No pesticides needed',
    'Tomato___Late_blight': 'Use Mancozeb or Ridomil',
    'Tomato___Leaf_Mold': 'Use Potassium bicarbonate or Neem oil'
}

def predict_and_recommend(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Make predictions
    predictions = model.predict(image)
    confidence = np.max(predictions[0])
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    
    # Debugging output with try-except to catch UnicodeEncodeError
    try:
        print(f"Image Path: {image_path}")
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
        print(f"Prediction Scores for Each Class: {predictions[0]}")  # See scores for all classes
    except UnicodeEncodeError as e:
        print(f"Encoding error occurred while printing: {e}")

    # Retrieve the recommendation based on the predicted class
    recommendation = pesticide_recommendations.get(predicted_class, "No recommendation available.")
    return predicted_class, recommendation, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the file in the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Use the relative path for the image URL
            image_url = url_for('static', filename='uploads/' + file.filename)

            # Make predictions
            disease, recommendation, confidence = predict_and_recommend(file_path)

            return render_template('index.html', disease=disease, recommendation=recommendation, confidence=confidence, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
