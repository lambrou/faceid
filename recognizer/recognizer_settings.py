from django.conf import settings
from django.contrib.staticfiles.templatetags.staticfiles import static

PREDICTOR_PATH = static('recognizer/bin/shape_predictor_5_face_landmarks.dat')
FACE_REC_MODEL_PATH = static('recognizer/bin/dlib_face_recognition_resnet_model_v1.dat')
FACES_FOLDER_PATH = static('recognizer/known_images/')