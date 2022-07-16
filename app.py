
from ctypes import sizeof
from flask import Flask, current_app, jsonify, send_file, request, session
app = Flask(__name__, static_url_path='')
from flask_cors import CORS

import io
import numpy as np
import sys

import pandas as  pd
from utlies import evaluate, to_image
import boto3

app = Flask(__name__)
CORS(app)

secret_key = "iuenp!m04*hu^@hieih" #secret_key for verifing backend requests
# (to be added to environment variables)

UPLOAD_FOLDER = 'static'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

tokens=[]

#storing tokens to txt file created on s3
S3_BUCKET = 'autodraw' 
folder = 'files/'
file_name = "stored_tokens.txt"
s3_access_key ='AKIAYHBCWEYTNPL6AYV4'
s3_secret_key='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'
s3_object= boto3.resource(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key
    )
s3 = boto3.client('s3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key)



def upload_prev_tokens():
    obj = s3.get_object(Bucket= S3_BUCKET, Key= folder+ file_name)
    stored_tokens=obj['Body'].read()
    global tokens
    stored_tokens =stored_tokens.decode("utf-8")
    tokens = set(stored_tokens.split("\n")) #["2", "3"]





@app.route('/insert_token', methods=['POST'])
def insert_token():
    global tokens
    data = request.get_json(force=True)
    token = data['token']
    try:
        key = data['secret_key']
    except:
        key=""
    if key == secret_key or "secret_key" in session:

        session["secret_key"] = "secret_key"

        upload_prev_tokens()
        tokens.add(token)
        tokens=list(tokens)
        s3.put_object(Body='\n'.join(tokens), Bucket=S3_BUCKET, Key='files/stored_tokens.txt')

        return jsonify(tokens), 201
    else:
        return jsonify({"response": "UNAUTHORIZED"}), 401 # UNAUTHORIZED



@app.route('/delete_token', methods=['Delete'])
def delete_token():
    global tokens
    data = request.get_json(force=True)
    token = data['token']
    try:
        key = data['secret_key']
    except:
        key=""
    if key == secret_key or "secret_key" in session:
        clear_session(token)
        upload_prev_tokens()
        if token in tokens:
            tokens.remove(token)
            tokens=list(tokens)
            s3.put_object(Body='\n'.join(tokens), Bucket=S3_BUCKET, Key='files/stored_tokens.txt')
        else:
            return jsonify({"response": "This token does not exist!"}) , 404 # NOT Found
        return jsonify(tokens)
    else:
        return jsonify({"response": "UNAUTHORIZED"}), 401 # UNAUTHORIZED


def clear_session(token):
    session.pop(token, None)


@app.route('/generate_image', methods = ['POST'])
def generate_image():
    print(f"Type of request.json: {type(request.json)}")# list has no attribute shape
    print(f"Shape of request.json: {len(request.json)}")# 
    input_data = request.get_json(force=True)
    labelmap = np.asarray(input_data["data"]) # labelmap ->[512,512]
    # labelmap = np.asarray(request.json) # labelmap ->[512,512]
    print(f"Type of labelmap: {type(labelmap)}")
    print(f"shape of labelmap: {labelmap.shape}")
    image = evaluate(labelmap) # image -> [1, 3, 512, 512]

    nmpy=image.numpy()
    print(f"output shape: {nmpy.shape}")

    #This part for writing generated numpy array in labelmap.txt file..
    np.set_printoptions(threshold=sys.maxsize)
    with open('labelmap.txt', 'w') as file:
        content = str(nmpy)
        file.write(content) 

    image = to_image(image)
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    image.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


@app.route('/generate_array', methods=['POST'])
def generate_array():
    global tokens
    input_data = request.get_json(force=True)
    labelmap = np.asarray(input_data["data"])
    token = input_data["token"]


    if token:
        
        upload_prev_tokens()
        if token in tokens:
            session[token] = token
        else:
            return jsonify({"response": "Invalid token"}) , 401
    elif "token" in session:
        token = session["token"]
    else:
        return jsonify({"response": "Token is required or session is expired"}), 401 # UNAUTHORIZED

    if not input_data["data"]:
        return jsonify({"response": "Image file is required"}), 404 # not found
    else:
        try:
            image = evaluate(labelmap) # image -> [1, 3, 512, 512]
            nmpy=image.numpy()
            return (str(nmpy))
        except:
            return jsonify({"response": "Image file is required"}), 400 # Bad Request
    



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')




