import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import get_file
from keras.utils.image_utils import img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pandas as pd
import pickle as pk
import cv2 as cv2
from keras.models import load_model
import matplotlib.pyplot as plt

first_gate = VGG16(weights='imagenet')

with open('./model/vgg16_cat_list.pk', 'rb') as f:
    cat_list = pk.load(f)
print("Cat list loaded")

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def car_categories_gate(img_224, model):
    print("Validating that this is a picture of your car...")
    out = model.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            return True
    return False


def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def predictor(image_path, csv_path, model_path, averaged=True, verbose=True):
    # read in the csv file
    class_df = pd.read_csv(csv_path)
    class_count = len(class_df['class'].unique())
    img_height = int(class_df['height'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = class_df['scale by'].iloc[0]
    image_list = []
    # determine value to scale image pixels by
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])
    path_list = []
    path_list.append(image_path)
    # paths = os.listdir(sdir)
    # for f in paths:
    #     path_list.append(os.path.join(sdir, f))
    if verbose:
        print(' Model is being loaded- this will take about 10 seconds')
    model = load_model(model_path)
    image_count = len(path_list)
    image_list = []
    file_list = []
    good_image_count = 0
    for i in range(image_count):
        try:
            img = cv2.imread(path_list[i])
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            good_image_count += 1
            img = img * s2 - s1
            image_list.append(img)
            file_name = os.path.split(path_list[i])[1]
            file_list.append(file_name)
        except:
            if verbose:
                print(path_list[i], ' is an invalid image file')
    if good_image_count == 1:  # if only a single image need to expand dimensions
        averaged = True
    image_array = np.array(image_list)
    # make predictions on images, sum the probabilities of each class then find class index with
    # highest probability
    preds = model.predict(image_array)
    if averaged:
        psum = []
        for i in range(class_count):  # create all 0 values list
            psum.append(0)
        for p in preds:  # iterate over all predictions
            for i in range(class_count):
                psum[i] = psum[i] + p[i]  # sum the probabilities
        index = np.argmax(psum)  # find the class index with the highest probability sum
        klass = class_df['class'].iloc[index]  # get the class name that corresponds to the index
        prob = psum[index] / good_image_count  # get the probability average
        # to show the correct image run predict again and select first image that has same index
        for img in image_array:  # iterate through the images
            test_img = np.expand_dims(img, axis=0)  # since it is a single image expand dimensions
            test_index = np.argmax(
                model.predict(test_img))  # for this image find the class index with highest probability
            if test_index == index:  # see if this image has the same index as was selected previously
                if verbose:  # show image and print result if verbose=1
                    plt.axis('off')
                    plt.imshow(img)  # show the image
                    print(f'predicted species is {klass} with a probability of {prob:6.4f} ')
                break  # found an image that represents the predicted class
        return klass, prob, img, None
    else:  # create individual predictions for each image
        pred_class = []
        prob_list = []
        for i, p in enumerate(preds):
            index = np.argmax(p)  # find the class index with the highest probability sum
            klass = class_df['class'].iloc[index]  # get the class name that corresponds to the index
            image_file = file_list[i]
            pred_class.append(klass)
            prob_list.append(p[index])
        Fseries = pd.Series(file_list, name='image file')
        Lseries = pd.Series(pred_class, name='species')
        Pseries = pd.Series(prob_list, name='probability')
        df = pd.concat([Fseries, Lseries, Pseries], axis=1)
        if verbose:
            length = len(df)
            print(df.head(length))
        return None, None, None, df


def get_result(klass, prob, data ,type):
    if 0 < prob < 30:
        result = {
            'error': None,
            'result': {
                'image_instrument': type,
                'probability': "{:.2f}".format(prob),
                'severity_level': 'minor',
                'price': data['data']['minor'][klass]
            }
        }
        return result
    elif 30 < prob < 70:
        result = {
            'error': None,
            'result': {
                'image_instrument': type,
                'probability': "{:.2f}".format(prob),
                'severity_level': 'moderate',
                'price': data['data']['moderate'][klass]
            }
        }
        return result
    else:
        result = {
            'error': None,
            'result': {
                'image_instrument': type,
                'probability': "{:.2f}".format(prob),
                'severity_level': 'major',
                'price': data['data']['moderate'][klass]
            }
        }
        return result


def calculate_price(klass, prob):
    data = []
    with open('./data/price_data.json', 'r') as f:
        data = json.load(f)

    if klass == "bumper_dent":
        return get_result(klass, prob, data , "Bumper Dent")
    elif klass == "bumper_scratch":
        return get_result(klass, prob, data , "Bumper Scratch")
    elif klass == "door_dent":
        return get_result(klass, prob, data , "Door Dent")
    elif klass == "door_scratch":
        return get_result(klass, prob, data , "Door Scratch")
    elif klass == "glass_shatter":
        return get_result(klass, prob, data , "Glass Shatter")
    elif klass == "head_lamp":
        return get_result(klass, prob, data , "Head Lamp")
    elif klass == "tail_lamp":
        return get_result(klass, prob, data ,"Tail Lamp")


def engine(img_path):
    csv_path = "./model/class_dict.csv"  # path to class_dict.csv
    model_path = "./model/EfficientNetB3-instruments-94.99.h5"

    img_224 = prepare_img_224(img_path)
    g1 = car_categories_gate(img_224, first_gate)

    if g1 is False:
        result = {
            'error': 'Are you sure this is a picture of your car? Please retry your submission.',
            'result': {
                'image_instrument': None,
                'probability': None,
                'severity_level ': None,
                'price': None
            }
        }
        return result
    else:
        klass, prob, img, df = predictor(img_path, csv_path, model_path, averaged=True, verbose=False)
        if klass == 'unknown':
            result = {
                'error': 'Are you sure this is a picture of a damage car? Please retry your submission.',
                'result': {
                    'image_instrument': None,
                    'probability': None,
                    'severity_level ': None,
                    'price': None
                }
            }
            return result
        else:
            return calculate_price(klass, prob*100)
