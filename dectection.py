import datetime
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
import geocoder
from pymongo import MongoClient
import json
from bson import ObjectId, json_util


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


first_gate = VGG16(weights='imagenet')

with open('./model/vgg16_cat_list.pk', 'rb') as f:
    cat_list = pk.load(f)
print("Cat list loaded")

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def get_database():
    # add db URL
    CONNECTION_STRING = "mongodb+srv://dbuser:12345@cluster0.rrmk7.mongodb.net/test?retryWrites=true&w=majority"

    client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    try:

        print(client.server_info())
        db = client['car_damage']
        collection = db['records']
        print("ss", db)
        return collection, False

    except Exception as err:
        print("ERROR : ", err)
        return err, True


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


def predictor(path_list, csv_path, model_path, averaged=False, verbose=True):
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
    # path_list = []
    # path_list.append(image_path)
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
    # if good_image_count == 1:  # if only a single image need to expand dimensions
    #     averaged = True
    image_array = np.array(image_list)
    # make predictions on images, sum the probabilities of each class then find class index with
    # highest probability
    preds = model.predict(image_array)
    # if averaged:
    #     psum = []
    #     for i in range(class_count):  # create all 0 values list
    #         psum.append(0)
    #     for p in preds:  # iterate over all predictions
    #         for i in range(class_count):
    #             psum[i] = psum[i] + p[i]  # sum the probabilities
    #     index = np.argmax(psum)  # find the class index with the highest probability sum
    #     klass = class_df['class'].iloc[index]  # get the class name that corresponds to the index
    #     prob = psum[index] / good_image_count  # get the probability average
    #     # to show the correct image run predict again and select first image that has same index
    #     for img in image_array:  # iterate through the images
    #         test_img = np.expand_dims(img, axis=0)  # since it is a single image expand dimensions
    #         test_index = np.argmax(
    #             model.predict(test_img))  # for this image find the class index with highest probability
    #         if test_index == index:  # see if this image has the same index as was selected previously
    #             if verbose:  # show image and print result if verbose=1
    #                 plt.axis('off')
    #                 plt.imshow(img)  # show the image
    #                 print(f'predicted species is {klass} with a probability of {prob:6.4f} ')
    #             break  # found an image that represents the predicted class
    #     return klass, prob, img, None
    # else:  # create individual predictions for each image
    data = []
    data_err = []
    total_price = 0
    err = False
    for i, p in enumerate(preds):
        index = np.argmax(p)  # find the class index with the highest probability sum
        klass = class_df['class'].iloc[index]  # get the class name that corresponds to the index
        image_file = file_list[i]
        if klass == "unknown":
            err = True
            data_err.append({'file_name': image_file})
            continue
        prob_obj, price = calculate_price(klass, p[index], image_file)
        data.append(prob_obj)
        total_price = total_price + price
    if err is True:
        return data_err, total_price, err
    return data, total_price, err


def get_result(klass, prob, data, type, image_file):
    if 0 < prob < 30:
        price = data['data']['minor'][klass]
        result = {
            'file_name': image_file,
            'image_instrument': type,
            'probability': "{:.2f}".format(prob),
            'severity_level': 'Minor',
            'price': price
        }
        return result, price
    elif 30 < prob < 70:
        price = data['data']['moderate'][klass]
        result = {
            'file_name': image_file,
            'image_instrument': type,
            'probability': "{:.2f}".format(prob),
            'severity_level': 'Moderate',
            'price': price
        }
        return result
    else:
        price = data['data']['major'][klass]
        result = {
            'file_name': image_file,
            'image_instrument': type,
            'probability': "{:.2f}".format(prob),
            'severity_level': 'Major',
            'price': price
        }
        return result, price


def calculate_price(klass, prob, image_file):
    with open('./data/price_data.json', 'r') as f:
        data = json.load(f)

    if klass == "bumper_dent":
        return get_result(klass, prob, data, "Bumper Dent", image_file)
    elif klass == "bumper_scratch":
        return get_result(klass, prob, data, "Bumper Scratch", image_file)
    elif klass == "door_dent":
        return get_result(klass, prob, data, "Door Dent", image_file)
    elif klass == "door_scratch":
        return get_result(klass, prob, data, "Door Scratch", image_file)
    elif klass == "glass_shatter":
        return get_result(klass, prob, data, "Glass Shatter", image_file)
    elif klass == "head_lamp":
        return get_result(klass, prob, data, "Head Lamp", image_file)
    elif klass == "tail_lamp":
        return get_result(klass, prob, data, "Tail Lamp", image_file)
    # elif klass == "unknown":
    #     return {'filename': image_file}


def engine(db , db_err):
    csv_path = "./model/class_dict.csv"  # path to class_dict.csv
    model_path = "./model/EfficientNetB3-instruments-94.99.h5"
    img_path = "./uploads/temp_image"

    response = {
        'created_at': None,
        'updated_at': None,
        'deleted_At': None,
        'error': None,
        'error_msg': None,
        'result': [],
        'calculated_price': None,
        'location': None,
        'time': None
    }

    timestamp = str(datetime.datetime.now())
    if db_err:
        response['error'] = True
        response['error_msg'] = 'Database Connection Error'
        return response

    path_list = []
    paths = os.listdir(img_path)
    for file in paths:
        path_list.append(os.path.join(img_path, file))

    for path in path_list:
        print(path)
        img_224 = prepare_img_224(path)
        g1 = car_categories_gate(img_224, first_gate)

        if g1 is False:
            fileName = os.path.basename(path)
            prop = {
                'file_name': fileName
            }
            response['error'] = True
            response['result'].append(prop)

    if response['error'] is True:
        response['created_at'] = timestamp
        response['updated_at'] = timestamp
        response['error_msg'] = 'Are you sure this is a picture of your car? Please retry your submission.'
        db.insert_one(json.loads(json_util.dumps(response)))

        return response, path_list

    klass, total_price, err = predictor(path_list, csv_path, model_path, averaged=True, verbose=False)

    if err is True:
        response['created_at'] = timestamp
        response['updated_at'] = timestamp
        response['error'] = True
        response['error_msg'] = 'Are you sure this is a picture of a damage car? Please retry your submission.'
        response['result'].extend(klass)
        db.insert_one(json.loads(json_util.dumps(response)))

        return response, path_list
    else:
        g = geocoder.ip('me')
        response['created_at'] = timestamp
        response['updated_at'] = timestamp
        response['location'] = {
            'address': g.address
        }
        response['error'] = 0
        response['result'].extend(klass)
        response['calculated_price'] = total_price
        response['time'] = timestamp
        db.insert_one(json.loads(json_util.dumps(response)))

        return response, path_list
