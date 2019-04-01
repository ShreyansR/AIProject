import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')                         # Use only for MAC OS
import matplotlib.pyplot as pyplot
from scipy.misc import toimage
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((4, 4), (i, j))
            pyplot.imshow(toimage(X[k]))
            k = k + 1
    # show the plot
    pyplot.show()


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# mean-std normalization
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

#show_imgs(x_test[:16])

# Load trained CNN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

image = load_img('bird.jpg', target_size=(32, 32))

image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("test")
#pred = model.predict(image)
#print('Predicted:', decode_predictions(pred, top=3)[0])
#np.argmax(pred[0])
indices = np.argmax(model.predict(image))
#indices = model.predict(image)
print(indices)
print(labels[indices])

#print('saving output to output.jpg')
#pred_img = image.array_to_img(indices)
#pred_img.save('output.jpg')