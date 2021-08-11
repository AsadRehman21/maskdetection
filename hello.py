import paho.mqtt.client as mqtt ;
from flask import Flask, render_template, Response , flash, request, redirect, url_for,session
import cv2
import face_recognition
import numpy as np
import os
import time
import hashlib
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import WebcamVideoStream
import imutils



import geocoder
from geopy.geocoders import Nominatim
from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip 
import timeit
import struct
import io
from array import array
app = Flask(__name__, static_url_path='/static')
app.secret_key = "super secret key"
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'asadali21'
app.config['MYSQL_DATABASE_DB'] = 'mask'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql = MySQL(app)
conn = mysql.connect()
cur =conn.cursor()
vs= WebcamVideoStream(src=0).start()



def detect_and_predict_mask(frame, faceNet, maskNet):
    
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	
	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
		
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	
	if len(faces) > 0:
	
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)



def gen_frames():  
        
                 
        prototxtPath = r"face_detector/deploy.prototxt"
        weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

      
        maskNet = load_model("mask_detector.model")

       
        print("[INFO] starting video stream...")
        
        

        while True:
    
            frame = vs.read()
            
            frame = imutils.resize(frame, width=400)

           
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

         
            for (box, pred) in zip(locs, preds):
               
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask on protection level : " if mask > withoutMask else "No Mask covid exposure level : "
                color = (0, 255, 0) if label == "Mask on protection level : " else (0, 0, 255)
                
             
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

       
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

   
        
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    
  
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/delete/<id>',methods = ['POST', 'GET'])
def delete(id):

    cur.execute("DELETE FROM Sign WHERE idSign = %s",(id))
    conn.commit()
    cur.execute("SELECT * FROM Sign")
    allUsers=cur.fetchall()
    return render_template('admin.html', result=allUsers)


@app.route('/accuracy')
def accuracy():
   
    
    cur.execute("SELECT * FROM stats")
    session['stats']=cur.fetchall()
    return render_template('accuracy.html',stats=session['stats'])

@app.route('/adminSing')
def adminSing():
   
    return render_template('adminSing.html')

@app.route('/admin' ,methods = ['POST', 'GET'])
def admin():
     if request.method == 'POST':
         username = request.form['username']
         password = request.form['password']
         
         count = cur.execute("SELECT * FROM admin where username = %s and pass=%s",(username,password))
        
         if count == 0:
           
             flash("Invalid Username and Password")
           
             return render_template('adminSing.html')
           
         else:
              
              data=cur.fetchone()
              session['id']=data[0]
              session['user']=data[1]
              cur.execute("SELECT * FROM Sign")
              session['allUsers']=cur.fetchall()
              
              return render_template('admin.html', result=session['allUsers'])
     return render_template('admin.html', result=session['allUsers'])
     
   
    

@app.route('/update' ,methods = ['POST', 'GET'])
def update():
    if request.method == 'POST':
         username = request.form['username']
         Email = request.form['Email']
         Phone = request.form['Phone']
        
         
         cur.execute("UPDATE Sign SET Username=%s, Email=%s, Phone=%s WHERE idSign=%s",(username, Email, Phone,session['id']))
         conn.commit()
         session['user']=request.form['username']
         session['email']=request.form['Email']
         session['phone']=request.form['Phone']
         flash("Updated Successfully..")
    return render_template('Profile.html')


@app.route('/Profile')
def Profile():
    
                  
    return render_template('Profile.html')
@app.route('/main')
def main():
       
    return render_template('main.html')


@app.route('/SignUp')
def SignUp():
 
    return render_template('SignUp.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
    
     if request.method == 'POST':
         username = request.form['username']
         Email = request.form['Email']
         Phone = request.form['Phone']
         password = request.form['password']
         
         count = cur.execute('select * from Sign where Email=%s', Email)  # prevent SqlInject

         if count == 0:
            cur.execute("INSERT INTO Sign (Username,Email,Phone,Pass)VALUES (%s,%s,%s,%s)", (username,Email,Phone,password))
            conn.commit()
         else:
        # the email exists
           flash("Email Already Exist..")
           
           return render_template('SignUp.html')
            
       
         
         return render_template('index.html')



@app.route('/login',methods = ['POST', 'GET'])
def login():
    
     if request.method == 'POST':
         username = request.form['username']
         password = request.form['password']
         
         count = cur.execute("SELECT * FROM Sign where Username = %s and Pass=%s",(username,password))
        
         if count == 0:
           
            flash("Invalid Username and Password")
           
            return render_template('index.html')
           
         else:
                prototxtPath = r"face_detector/deploy.prototxt"
                weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
                faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

          
                maskNet = load_model("mask_detector.model")

                print("[INFO] starting video stream...")
                
                time.sleep(3.0)
                maskcheck=False
           
                while True:
                  
                    frame = vs.read()
                    frame = imutils.resize(frame, width=400)

                  
                    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                    
                    for (box, pred) in zip(locs, preds):
                     
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        label = "Mask" if mask > withoutMask else "No Mask"
                        
                        if label == "Mask":
                            
                            color = (0, 255, 0) 
                            maskcheck=True
                            break
                        else:
                            
                            (0, 0, 255)
                    if maskcheck==True:
                        break
                            
                        
                    
                        
                        
                        
                        
                        
                        
                        
                        
                if maskcheck==True:
                    
                    
                    data=cur.fetchone()
                    session['id']=data[0]
                    session['user']=data[1]
                    session['email']=data[2]
                    session['phone']=data[4]
                    now = datetime.now()

                    current_time = now.strftime("%H:%M:%S")
                    cur.execute("INSERT INTO stats (username,maskAccuracy,LastLogIn)VALUES (%s,%s,%s)", (session['user'],max(mask, withoutMask) * 100,current_time))
                    conn.commit()
                    
                    return render_template('main.html')
                else:
                    flash("Face MAsk Not Detetcted")
            
                    return render_template('index.html')
                  
   
           
            
       
         
         

if __name__ == '__main__':
   app.run()