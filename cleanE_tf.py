#@Name: Clean Energy Application
#@Author: Yinng Qin

import os

# https://keras.io/api/applications/
# using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from pickle import dump

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50



def __init_inception():
    # google inception_v3 model and microsoft resNet50 model
    model = InceptionV3()
    # summarize the model
    model.summary()
    return model

def __init__resnet():
    # loading the resnet50 model
    model = ResNet50()
    # summarize the model
    model.summary()
    print('\n')
    return model

def _init__vgg16():
    # load the vgg16 model
    model = VGG16()
    # summarize the model
    model.summary()
    return model

def _init__MobileNet():
    model = MobileNet()
    model.summary()
    return model

def build_transfer_model():
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

def extracted_features(model, img_name,img_ori):
    # load an image from file
    image = load_img(img_name, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # get extracted features
    features = model.predict(image)
    print(features.shape)
    # save to file
    save_file = 'data/pkl/' + img_ori + '_transfer_model.pkl'
    dump(features, open(save_file, 'wb'))
    

def predict_img(model, img_name):
    print('....predict......................................................\n')
    # load an image from file
    image = load_img(img_name, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    # print('%s (%.4f%%)' % (label[1], label[2]*100))
    return label



if __name__ == '__main__':
    
    
    test_dir = 'data/test/'
    for img_name in os.listdir(test_dir):
        model = _init__vgg16()
        # model = __init_inception()
        img_ori = img_name
        img_name = test_dir + img_name
        label = predict_img(model, img_name)
        print(img_name, '%s (%.4f%%)' % (label[1], label[2]*100))
        print('\n.................................................................\n')
    
        model = build_transfer_model()
        extracted_features(model, img_name,img_ori)
    
    
    
