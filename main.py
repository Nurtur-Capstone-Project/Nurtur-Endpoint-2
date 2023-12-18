from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load pre-trained model
model = load_model('ResNet50V2_Model.h5')


@app.route('/', methods=['GET'])
def home():
    return 'berhasil,hore!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_file = request.files['image']

        # Read image file and convert to RGB
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)

        # Resize image to match model's expected sizing
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0) / 255.0

        # Perform prediction
        result = model.predict(image)
        emotion_class = np.argmax(result)

        # Map class index to emotion label
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = emotion_labels[emotion_class]

        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)