from flask import Flask, render_template, Response
import cv2


face_cascade = cv2.CascadeClassifier("D:/OPENCV_HTML/resources/haarcascade_frontalface_default.xml")

app = Flask(__name__)


def faceDetection():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)  # use 0 for web camera

    while True:
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame = cv2.imencode(".jpg", img)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(faceDetection(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
