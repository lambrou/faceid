from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from django.conf.urls.static import static
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import cv2
import base64
import numpy as np
import dlib
import time
import os
import os.path

os.makedirs(settings.UNKNOWN_FACES_PATH, exist_ok=True)
os.makedirs(settings.FACES_FOLDER_PATH, exist_ok=True)

# Create camera class
class Camera(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    def __del__(self):
        self.cam.release()

    # Capture the image
    def get_frame(self):
        output = []
        ret, image = self.cam.read()
        if not ret:
            print("Image capture failure")
        # Encode the image
        ret, jpg = cv2.imencode('.jpg', image)
        output.append(jpg)
        output.append(image)
        return output

def capture(camera):
    return camera.get_frame()

def dlibVect_to_numpyNDArray(vector):
    array = np.zeros(shape=128)
    for i in range(0, len(vector)):
        array[i] = vector[i]
    return array

# Subtract the unknown (unauthorized) face data array from the known array
# and then return the result. The result is known as the 'Euclidian Distance'
def get_euc_dist(known, unknown):
    npknown = dlibVect_to_numpyNDArray(known)
    npunknown = dlibVect_to_numpyNDArray(unknown)
    dist = np.linalg.norm(npknown - npunknown)
    return dist

def get_face_desc(image):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(settings.PREDICTOR_PATH)
    facerec = dlib.face_recognition_model_v1(settings.FACE_REC_MODEL_PATH)

    img = dlib.load_rgb_image(image)


    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return face_descriptor
  
def index(request):
    if request.method == 'POST':

        if 'file' in request.FILES:
  
            if request.FILES['file']:
                try:
                    os.remove(settings.FACES_FOLDER_PATH + 'img.jpg')
                except OSError:
                    pass
                file = request.FILES['file']
                fs = FileSystemStorage()
                filename = fs.save(settings.FACES_FOLDER_PATH + 'img.jpg', file)
        
        if 'img' in request.POST:
            img2 = False
            match = False
            camObj = Camera()
            img = capture(camObj)
            
            temp = cv2.imwrite(settings.UNKNOWN_FACES_PATH + 'temp.jpg', img[1])

            b64 = base64.b64encode(img[0])
            img_str = b64.decode('ascii')


            known = get_face_desc(settings.FACES_FOLDER_PATH + 'img.jpg')
            unknown = get_face_desc(settings.UNKNOWN_FACES_PATH + 'temp.jpg')
            if os.path.isfile(settings.UNKNOWN_FACES_PATH + 'temp.jpg'):
                img2 = True 
            if unknown and known:
                distance = get_euc_dist(known, unknown)
                if distance < settings.COMPARISON_THRESHOLD:
                    match = True
     

            return render(request, 'recognizer/index.html', { 'img': img_str, 'img2': img2, 'match': match })
    return render(request, 'recognizer/index.html', None)