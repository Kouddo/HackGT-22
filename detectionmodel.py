import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))


def create_data(csv):
    testing = []
    for i in csv['filename']:
        image = cv2.imread('/Users/achennupati/Desktop/ocular/preprocessed_images/' + i)/255
        testing.append(image)
    labels = csv['labels']
    labels = [labeldic[labels[i][2]] for i in range(len(testing))]
    print("data created")

    return testing, labels


labeldic = {'N': 1, 'D': 2, 'N': 3, 'M': 4, 'O': 5, 'H': 6, 'C': 7, 'A': 8, 'G': 9}


testing, labels = create_data(pd.read_csv('/Users/achennupati/Desktop/ocular/full_df.csv'))

train_images,train_labels = np.asarray(testing[0:1000]), np.asarray(labels[0:1000])

test_images, test_labels = np.asarray(testing[6000:6392]), np.asarray(labels[6000:6392])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
history = model.fit(train_images, train_labels, epochs=1, 
                    validation_data=(test_images, test_labels))

print(model.evaluate(test_images,  test_labels, verbose=2))
"""