# import the opencv library
import os
import cv2
import sys
from flask import Flask, render_template, Response
from crap_code.image import CameraSwap

face_path = sys.argv[2]
if not os.path.exists(face_path):
    print(f"Face {face_path} not found")
    sys.exit(1)

app = Flask(__name__)

# define a video capture object
swapper = CameraSwap(int(sys.argv[1]), sys.argv[2], profile=True)


def next_frame():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        frame = swapper.get_frame()  # read the camera frame
        if frame is None:
            break
        else:
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )  # concat frame one by one and show result


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(next_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
