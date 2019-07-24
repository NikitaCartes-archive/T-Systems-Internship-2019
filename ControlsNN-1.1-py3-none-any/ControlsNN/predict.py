# import the necessary packages

# this package is required to work with binary labels
import pickle
# this package is required to load and display images
import cv2
# this function that loads the neural network model
from keras.models import load_model
# this package is required to load NN
import os
import inspect
import numpy as np


def predict_image(image_path):
    # this function loads an image from a specified path
    # and determines which control is shown

    # load the image and resize it
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, (64, 64))

    # scale the pixel values from [0, 255] to [0, 1]
    image = image.astype("float") / 255.0

    # add the batch dimension
    image = image.reshape((1, image.shape[0], image.shape[1],
            image.shape[2]))

    # load model and label binarizer
    model = load_model(os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+ "/vggnet.model")
    lb = pickle.loads(open(os.path.abspath(os.path.dirname(os.path.abspath(inspect.stack()[0][1])) +"/vggnet_lb.pickle"), "rb").read())

    # make a prediction of what is depicted in the image
    preds = model.predict(image)

    # find the most exact match with the data
    # on which the neural network was trained
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    #show the image with the accuracy defined by the neural network
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 0, 255), 1)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    
def predict_images(images_path):
    # this function loads images from a specified path
    # and determines which controls is shown

    # get files from directory
    files_name = os.listdir(images_path)
    
    images = []
    
    for f in files_name:
        images.append(cv2.resize(cv2.imread(images_path + f), (64, 64)))

    for i in range(len(images)):
        images[i] = images[i].astype("float")/255
        
    # add the batch dimension
    images = np.array(images).reshape((-1, images[0].shape[0], images[0].shape[1], images[0].shape[2]))

    # load model and label binarizer
    model = load_model(os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+ "/vggnet.model")
    lb = pickle.loads(open(os.path.abspath(os.path.dirname(os.path.abspath(inspect.stack()[0][1])) +"/vggnet_lb.pickle"), "rb").read())

    # make a prediction of what is depicted in the image
    predictions = model.predict(images)

    return list(zip(files_name, predictions))
    
