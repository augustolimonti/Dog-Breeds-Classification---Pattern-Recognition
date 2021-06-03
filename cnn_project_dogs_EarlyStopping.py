import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.models import model_from_json
import csv
print(tf.__version__)

training_set = pd.read_csv("smaller_train.csv")

training_imgs = ["Dog_Images/{}".format(x) for x in list(training_set.Images)]

print(len(training_set))

training_labels_1 = list(training_set['Breed'])

training_set = pd.DataFrame( {'Images': training_imgs,'Breed': training_labels_1})

#Changing the type to str
training_set.Breed = training_set.Breed.astype(str)
print(training_set.head())

train_dataGen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
                                    zoom_range = 0.2, horizontal_flip = True)


train_generator = train_dataGen.flow_from_dataframe(dataframe = training_set,
                        directory = "", x_col = "Images", y_col = "Breed", class_mode = "categorical",
                        target_size = (224,224), batch_size = 4000)

x, y = train_generator.next()

classifier = Sequential()

#First Convolutional layer
classifier.add(Conv2D(filters = 32,kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#second Convolutional layer
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#third Convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


#Flattening
classifier.add(Flatten())

#Hidden Layer
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))

#Output Layer
classifier.add(Dense(units = 30, activation = 'softmax'))
# classifier.add(Dense(units = 30, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                    metrics = ['val_accuracy','accuracy'])
#
# es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)

classifier.summary()

# step_size_train = x.n//x.batch_size
classifier.fit(x, y, epochs = 50, validation_split = 0.2)
# classifier.fit(train_generator, epochs = 20, steps_per_epoch = step_size_train)

#serialize model to JSON
class_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(class_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")

# load json and create model
json_file = open('classifier.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = tf.keras.models.model_from_json(loaded_classifier_json)
# load weights into new model
loaded_classifier.load_weights("classifier.h5")
print("Loaded model from disk")

test_set = pd.read_csv("smaller_test.csv")
test_imgs = ["Dog_Images/{}".format(x) for x in list(test_set.Images)]
test_set = pd.DataFrame( {'Images': test_imgs })

classes = train_generator.class_indices
print("classes: ", classes)

inverted_classes = dict(map(reversed, classes.items()))
print("inverted classes: ", inverted_classes)

from keras.preprocessing import image

Y_pred = []
#
for i in range(0, len(test_set)):
  # print(test_set.Images[i])
  img = image.load_img(path = test_set.Images[i],target_size=(224,224,3))
  img = image.img_to_array(img)
  test_img = img.reshape((1,224,224,3))
  img_class = loaded_classifier.predict_classes(test_img)
  prediction = img_class[0]
  Y_pred.append(prediction)

print(Y_pred)
prediction_classes = [inverted_classes.get(item,item) for item in Y_pred]
print(prediction_classes)
print(len(prediction_classes))

breed = []
name = []
BASE_DIR = "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Dataset"
folders = os.listdir(BASE_DIR)
folders.remove(".DS_Store")
index = 0

for i in prediction_classes:
  breed.append(i[0:])
  for files in folders:
      label = folders.index(files)
      if breed[index] == str(label):
          name.append(files)
          index = index + 1
          break

with open("test_results.csv", 'r') as f:
    total = 0
    count = 0
    csv_input = csv.DictReader(f)
    for row in csv_input:
        if row["Breed"] == breed[total]:
            count = 1 + count
        total = 1 + total

print(count/total)

predictions = {}
predictions['Breed'] = breed
predictions['Name'] = name

#Writing to excel
pd.DataFrame(predictions).to_excel("predictions.xlsx", index = False)
