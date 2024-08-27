   from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


classifier = Sequential()
                                                 
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten()) 

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) 


(classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']))


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/val',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

csv_logger=tf.keras.callbacks.CSVLogger('metrics.csv')

classifier.fit_generator(training_set,
                         steps_per_epoch = 85, # number of training images
                         epochs =200,        
                         validation_data = test_set,
                         validation_steps = 85, #number of test images
                         callbacks=[csv_logger])  


#Testing the model 

classifier.save('signature_forgery_model3.h5')



fig = plt.figure(0)


var=pd.read_csv('metrics.csv')
print(var)
x = var['epoch']
e1=var['accuracy']
e2=var['loss']
y1=list(e1)
y2=list(e2)
plt.plot(x,y1,color='g',linestyle='dashed',marker='o',label="accuracy")
plt.plot(x,y2,color='r',linestyle='dashed',marker='o',label="loss")
plt.xlabel('epoch')
plt.ylabel('Accuracy and loss')
plt.title('Signature Forgery cases')
plt.legend()
plt.show()

fig.savefig('graph.jpg')



plt.close()





