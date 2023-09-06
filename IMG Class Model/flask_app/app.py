

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request

from PIL import Image


app = Flask(__name__)


#class label names
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]



#loads saved/trained model
model = tf.keras.models.load_model('saved_model/my_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handles the image upload
        image = request.files['image']
        image = Image.open(image)

        # Convert RGBA image to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predicts using the model
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions)
        class_name = class_labels[class_index]  
        
        return render_template('index.html', prediction=class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
