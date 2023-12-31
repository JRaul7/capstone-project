# -*- coding: utf-8 -*-
"""predict_capstone.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yZeOwcGysvijkvX673Q4RbJ_of0Rag9l
"""

from flask import Flask, request, jsonify
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf

labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

app = Flask('char_recognition')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    img_path = 'image.png'  # Saving the file temporarily
    file.save(img_path)

    # Load and prepare the image
    img = load_img(img_path, grayscale=True, target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='cnn_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predict_value = interpreter.get_tensor(output_details[0]['index'])

    digit = argmax(predict_value) - 10
    prediction = labels[digit]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)