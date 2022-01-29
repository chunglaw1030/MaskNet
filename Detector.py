# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# %%
RED = (0, 0, 255)
GREEN = (0, 255, 0)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') 

# %%
model_name = 'MaskNet.h5'
face_classifier = tf.keras.models.load_model('model/MaskNet')
class_names = ['mask', 'no_mask']

# %%
def get_extended_image(img, x, y, w, h, k=0.1):

    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                     start_x:end_x]
    face_image = tf.image.resize(face_image, [128, 128])
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

# %%
video_capture = cv2.VideoCapture(0)  # webcamera
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:

        face_image = get_extended_image(frame, x, y, w, h, 0.5)
        result = face_classifier.predict(face_image)
        prediction = tf.nn.sigmoid(result)
        if prediction < 0.5:
            prediction = 0
        else:
            prediction = 1

        prediction = class_names[prediction]
        confidence = np.array(prediction[0]).max(axis=0)  # degree of confidence

        if prediction == 'mask':
            color = GREEN
        else:
            color = RED
        cv2.rectangle(frame,
                      (x, y), 
                      (x+w, y+h), 
                      color,
                      2)  
        cv2.putText(frame,
                    # text to put
                    "{:6} - {:.2f}%".format(prediction, confidence*100),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,                      2,  
                    color,
                    2) 

    cv2.imshow("Face detector - to quit press ESC", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27: 
        break


# when everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Detortor stopped")

# %%



