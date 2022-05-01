import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing import image
import numpy as np

from google.colab import drive
 
drive.mount("/content/gdrive/")
train_dir  = '/content/gdrive/MyDrive/Deep Learning/Cats vs Dogs/training'
test_dir   = '/content/gdrive/MyDrive/Deep Learning/Cats vs Dogs/testing'
import tensorflow as tf
 
model  = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28, 3)),
        #tf.keras.layers.Dense(1000,activation=tf.nn.relu),#hidden
        #tf.keras.layers.Dense(500,activation=tf.nn.relu),
        #tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(20,activation=tf.nn.relu),
        tf.keras.layers.Dense(2,activation=tf.nn.softmax) # output layers
])
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics =['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
import os 
os.listdir(train_dir)
os.listdir(train_dir+'/cats')
batch_size = 10
 
# ကိုယ်ယူတဲ့ Target Size တွေကို မှတ်ထားပါ နောက်ပြန်သုံးမှာမို့
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(28, 28),
                                                 batch_size=batch_size,
                                                 #class_mode='categorical')
                                                 class_mode='binary')
 
test_set = test_datagen .flow_from_directory(test_dir,
                                            target_size=(28, 28),
                                            batch_size=batch_size,
                                            #class_mode='categorical')
                                            class_mode='binary')
history = model.fit(training_set,
                              #validation_data = validation_set,
                              steps_per_epoch=11//batch_size,
                        
                         epochs=10,
                         #validation_steps=validation_length//batch_size,
                         #validation_steps=1000//batch_size,
                         verbose = 2,
                         shuffle = False)
classes = model.predict(test_set)
folder_names = ['cats','dogs']
y_pred = []
for folder in folder_names:
    path = test_dir+"/"+folder
    path_fnames = os.listdir(path)
    for i in path_fnames:
        path2 = path+'/'+i
        img = tf.keras.preprocessing.image.load_img(path2, target_size=(28,28))#target size ကို သတိထားပါ
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x /= 255.0
        images = np.vstack([x])# [1 2 3 4 5 6]
        classes = model.predict(x)
        y_classes=classes.argmax(axis=-1)
        y_pred.append(y_classes[0])
    print()
    y_true = test_set.classes.tolist()
print(len(y_true))
lass_dictionary = test_set.class_indices
print('Labels dictionary',class_dictionary)
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print('Accuracy Score',accuracy_score(y_true, y_pred)*100,'%')
print('Precision Macro Score ',precision_score(y_true, y_pred,average = 'macro')*100,'%')
print('Recall_Score',recall_score(y_true, y_pred, average = 'macro')*100,'%')
print('F1_Score',f1_score(y_true, y_pred, average = 'macro')*100,'%')
model.save('/neural_networks.h5')
