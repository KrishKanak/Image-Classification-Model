import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator





#loading dataset CIFAR-10
(x_train,y_train), (x_test, y_test)= tf.keras.datasets.cifar10.load_data()

#prints the sahpes for the dataset

print("Training data shape", x_train.shape)
print("Training lables shape", y_train.shape)


print("Testing data shape", x_test.shape)


print("Testing lables shape", y_test.shape)



#print statement for number of classes


num_classes = len(set(y_train.flatten()))

print("Num of classes ", num_classes)



#pixel values from [0,1]

x_train = x_train.astype('float32')/255.0

x_test = x_test.astype('float32')/255.0



#class labels


y_train = tf.keras.utils.to_categorical(y_train, num_classes)

y_test = tf.keras.utils.to_categorical(y_test, num_classes)





from keras_preprocessing.image import ImageDataGenerator


#DATA AUGMENTATION



# Create a data generator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=15,      # Random rotation within the range of [-15, 15] degrees
    width_shift_range=0.1,  # Random horizontal shift within 10% of image width
    height_shift_range=0.1, # Random vertical shift within 10% of image height
    horizontal_flip=True,   # Randomly flip images horizontally
)

# Fit the data generator to the training data
datagen.fit(x_train)




 

