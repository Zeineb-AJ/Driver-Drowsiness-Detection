from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

app = Flask(__name__)
socketio = SocketIO(app)



mixer.init()
sound = mixer.Sound('C:/Users/User/Drowsiness detection/alarm.wav')

face_cascade_path = 'C:/Users/User/Drowsiness detection/haar cascade files/haarcascade_frontalface_alt.xml'
leye_cascade_path = 'C:/Users/User/Drowsiness detection/haar cascade files/haarcascade_lefteye_2splits.xml'
reye_cascade_path = 'C:/Users/User/Drowsiness detection/haar cascade files/haarcascade_righteye_2splits.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
leye_cascade = cv2.CascadeClassifier(leye_cascade_path)
reye_cascade = cv2.CascadeClassifier(reye_cascade_path)

lbl_values = ['Closed', 'Open']
model = load_model('C:/Users/User/Drowsiness detection/modeldrows.h5', compile=False)
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

@app.route('/detection')
def detection():
    return render_template('detection.html')

@socketio.on('detect_button')
def detect_button():
    global count, score, thicc
    pred = [99]
    lpred = [99]

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye = reye_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count += 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred_prob = model.predict(r_eye)
            rpred = np.argmax(rpred_prob, axis=1)
            predicted_label = lbl_values[rpred[0]]
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count += 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred_prob = model.predict(l_eye)
            lpred = np.argmax(lpred_prob, axis=1)
            predicted_label = lbl_values[lpred[0]]
            break

        if lpred[0] == 0 and rpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Stop the sound if both eyes are open
            if lpred[0] == 1 and rpred[0] == 1:
                try:
                    mixer.stop()
                except:
                    pass

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 15:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass
            if thicc < 16:
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources quand la boucle est terminée
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    socketio.run(app)