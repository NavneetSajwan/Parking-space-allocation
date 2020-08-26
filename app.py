#rahul pandit
#!/usr/bin/env python 
from flask import *
from flask import Flask, request, jsonify, json, make_response, redirect, session, send_from_directory
# import MySQLdb
from werkzeug.security import generate_password_hash, check_password_hash
import model
from datetime import timedelta #for time to define how long session is active
import os
#import magic
import urllib.request
#from app import app
from werkzeug.utils import secure_filename
import cv2
import secrets
import random
import string 
import random 
import base64
from io import BytesIO
from PIL import Image
from base64 import b64decode
import os.path
import cv2
import model
from time import sleep
#from flask_cors import CORS

########################  import function define in model file  #########

#from model import tyre_classification

#create session in flask 
#https://www.youtube.com/watch?v=iIhAfX4iek0
#https://techwithtim.net/tutorials/flask/sessions/

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

UPLOAD_FOLDER = os.getcwd() + '/static/uploads/'

app = Flask(__name__, static_url_path='')
#app.secret_key = secrets.token_urlsafe(16)
#CORS(app)


app.permanent_session_lifetime = timedelta(days=5)

app.secret_key = "hello" #encrypt data by this key on server
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024








######################################
"""@app.route('/static/<path:path>')
def send_js(path):
	return send_from_directory('uploads', path)


"""
#####################


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

@app.route("/")
def index():
	#if "email" in session:
	#	return redirect(url_for("upload"))
	#else:		
	return render_template("upload.html", title="Flask signUp form")


@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'files[]' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	
	files = request.files.getlist('files[]')
	# sleep(10)

	errors = {}
	success = False
	#i am return image shape
	result=[]
	for file in files:
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			img = cv2.imread(app.config['UPLOAD_FOLDER']+filename) #this code read image from folder
			#pred=tyre_classification(img)
			predictor, cfg = model.setup_model()
			print("predictor generated")
			torch_bbox = model.generate_label_bboxes()
			torchint_preds = model.gen_bbox_predictions(img, predictor)
			image_out = model.draw_output(torchint_preds, torch_bbox, img)
			name = filename.split(".")[:-1][0]
			ext = filename.split(".")[-1]
			file_name=name+'_final_.'+ext
			print(file_name)
			cv2.imwrite(app.config['UPLOAD_FOLDER']+file_name,image_out)
			result.append(1)
			print(img.shape)
			success = True
			resp={'message': " ",'success': True, 'image':'http://0.0.0.0:4000/uploads/'+file_name}
			print(resp['image'])
			return make_response(jsonify(resp))
		else:
			errors[file.filename] = 'File type is not allowed'
	
	if success and errors:
		errors['message'] = 'File(s) successfully uploaded'
		resp = jsonify(errors)
		resp.status_code = 206
		return resp
	if success:		
		if int(result[0])==1:
			result=[]
			result.append('Good Quality')
		else:
			result=[]
			result.append('puncture tyre')
		resp = jsonify({'message' :result[0]})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify(errors)
		resp.status_code = 400
		return resp
"""
***************stop ******************
"""




if __name__ == "__main__":
	app.run(debug=True,port=4000,host='0.0.0.0')




