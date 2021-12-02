import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from PIL import Image
import random
import itertools
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt  
%matplotlib inline
import os
os.chdir('...path...')
img_folder=r'.../data/ECG Data/'
test_folder=r'.../data/ECG Data/Resize_CovidECG250'

def create_dataset_PIL(test_folder):
    
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= np.array(Image.open(image_path))
            image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
            image = image.astype(int)
            # image /= 255  
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array , class_name

#Show data from test folder
plt.figure(figsize=(20,20))
for i in range(10):
    file = random.choice(os.listdir(test_folder))
    image_path= os.path.join(test_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,10,i+1)
    ax.title.set_text(file)
    plt.imshow(img)

#Input size
IMG_WIDTH=200
IMG_HEIGHT=120

#Array of images and class name
PIL_img_data, class_name = create_dataset_PIL(img_folder)

#Change class name to number
target_dict={k: v for v, k in enumerate(np.unique(class_name))}
target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
print(target_val.count(0)) #Covid
print(target_val.count(1)) #Normal
target_val = np.asarray(target_val)
PIL_img_data = np.asarray(PIL_img_data)

train_images,test_images,train_labels,test_labels = train_test_split(PIL_img_data,target_val,test_size=0.3,random_state=2)

class_names = ['ECG Images of COVID-19 Patients (250)','Normal Person ECG Images (859)']

#CNN
num_classes = 2 #output 2 nodes
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(120, 200, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(num_classes, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#Training
epochs=10
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(test_images)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


target_names = ['Covid','Normal']

y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix')

cm = confusion_matrix(test_labels, y_pred)
# plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

print('Classification Report')
print(classification_report(test_labels, y_pred, target_names=target_names))


ax= plt.subplot()
sns.heatmap(cm, annot=True ,cmap='YlGnBu',fmt='d')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Covid', 'Normal']);
ax.yaxis.set_ticklabels(['Covid', 'Normal']);