import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('/Users/krishkanak/IMG Class Website/saved_model')  # this can be replacable with a different image using the correct path 

# Dataset labels
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load and preprocess the new image
image_path = '/Users/krishkanak/Downloads/carimg.jpeg'  # this can be replacable with a different image using the correct path 
image = Image.open(image_path)

# Ensure the image has 4 channels (RGBA)
if image.mode != 'RGBA':
    image = image.convert('RGBA')

image = image.resize((32, 32))  # Resize the image to match the input size of the model
image_array = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]

# Ensure the image has the correct shape 
if image_array.shape[-1] != 4:
    raise ValueError("Input image does not have 4 channels (RGBA)")


# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)  

# Make predictions using the model
predictions = model.predict(image_array)
class_index = np.argmax(predictions)
class_name = class_labels[class_index]

print(f'The predicted class is: {class_name}')
