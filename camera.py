# import the opencv library
import os
import cv2
import sys
import time
from threading import Thread
from flask import Flask, render_template, Response
from crap_code.image import CameraSwap

face_path = sys.argv[2]
if not os.path.exists(face_path):
    print(f"Face {face_path} not found")
    sys.exit(1)

app = Flask(__name__)

# define a video capture object
# https://github.com/dorssel/usbipd-win/wiki/WSL-support
# usbipd attach --wsl --busid 5-2
# usbipd detach --busid 5-2
swapper = CameraSwap(int(sys.argv[1]), sys.argv[2], profile=True)
swapper.start()


def next_frame():  # generate frame by frame from camera
    while True:
        time.sleep(0.01)
        if swapper.output_frame is None:
            continue
        else:
            _, buffer = cv2.imencode(".jpg", swapper.output_frame)
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
    app.run(debug=False, port=8080)
