import os
from flask import Flask, request, json
from werkzeug.utils import secure_filename
from flask_cors import CORS
import dectection
from pymongo import MongoClient

uploads = './uploads/temp_image'

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = uploads

# Allowed file type
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


def get_database():
    # add db URL
    CONNECTION_STRING = "ADD UR MOGODB URI"

    client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    try:

        print(client.server_info())
        db = client['car_damage']
        collection = db['records']
        return collection, False

    except Exception as err:
        print("ERROR : ", err)
        return err, True


db, db_err = get_database()


# Check file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def response_config(data, status_code, mime_type):
    return app.response_class(
        response=json.dumps(data),
        status=status_code,
        mimetype=mime_type
    )


# Response internal server errors
def internal_server_error(e):
    data = {
        'Internal server error ': str(e)
    }
    return response_config(data, 500, 'application/json')


app.register_error_handler(500, internal_server_error)


@app.route('/api/detect-car-damage', methods=['POST'])
def get_fer_demography():
    print(request.files)
    if 'file' not in request.files:
        data = {
            'Message': 'Image not found!'
        }
        return response_config(data, 404, 'application/json')

    for image in request.files.getlist('file'):
        print(image)
        if image.filename == '':
            data = {
                'Message': 'No selected image!'
            }
            return response_config(data, 404, 'application/json')

        if image and allowed_file(image.filename):
            try:
                filename = secure_filename(image.filename)
                temp_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(temp_image)


            except Exception as e:
                app.logger.error('Error : ', str(e))

                data = {
                    'Message': str(e)
                }
                return response_config(data, 505, 'application/json')

        else:
            data = {
                'Message': 'Invalid image type!'
            }
            return response_config(data, 404, 'application/json')

    res, path_list = dectection.engine(db , db_err)
    try:
        for path in path_list:
            if os.path.isfile(path):
                os.remove(path)
            else:
                app.logger.warn("Error: %s file not found" % path)
    except Exception as e:
        app.logger.error('Error : ', str(e))

    return response_config(res, 202, 'application/json')


if __name__ == '__main__':
    app.run(debug=True)
