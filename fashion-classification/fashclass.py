import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def plot_image(i, preds, truths, img):
  tru, img = truths[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  pred = np.argmax(preds)
  if pred == tru:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[pred],
                                100*np.max(preds),
                                class_names[tru]),
                                color=color)

def plot_value_array(i, preds, truths):
  tru = truths[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  pl = plt.bar(range(10), preds, color="#777777")
  plt.ylim([0, 1])
  pred = np.argmax(preds)

  pl[pred].set_color('red')
  pl[tru].set_color('blue')

'''---'''

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-Shirt', 'Pants', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#last try: 0.88

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
start = np.random.randint(len(test_labels) - num_images)
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(start+i, predictions[start+i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(start+i, predictions[start+i], test_labels)
plt.tight_layout()
plt.show()
