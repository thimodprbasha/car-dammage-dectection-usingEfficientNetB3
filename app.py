import os
from uuid import uuid4

from flask import Flask, request, json
from werkzeug.utils import secure_filename
import  dectection
uploads = './uploads/temp_image'

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = uploads

# Allowed file type
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Check file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def response_config(data, status_code, mime_type):
    return application.response_class(
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


application.register_error_handler(500, internal_server_error)


@application.route('/api/detect-car-damage', methods=['POST'])
def get_fer_demography():
    if 'file' not in request.files:
        data = {
            'Message': 'Image not found!'
        }
        return response_config(data, 404, 'application/json')

    image = request.files['file']
    if image.filename == '':
        data = {
            'Message': 'No selected image!'
        }
        return response_config(data, 404, 'application/json')

    if image and allowed_file(image.filename):
        try:
            filename = secure_filename(image.filename)
            temp_image = os.path.join(application.config['UPLOAD_FOLDER'], uuid4().__str__() + filename)
            image.save(temp_image)

            res = dectection.engine(temp_image)

            if os.path.isfile(temp_image):
                os.remove(temp_image)
            else:
                application.logger.warn("Error: %s file not found" % temp_image)

            return response_config(res, 202, 'application/json')

        except Exception as e:
            application.logger.error('Model Error : ', str(e))

            data = {
                'Message': str(e)
            }
            return response_config(data, 505, 'application/json')

    else:
        data = {
            'Message': 'Invalid image type!'
        }
        return response_config(data, 404, 'application/json')


if __name__ == '__main__':
    application.run(debug=True)
