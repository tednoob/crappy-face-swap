# import the opencv library
import os
import cv2
import sys
from flask import Flask, render_template, Response
from crap_code.image import FaceSwap

app = Flask(__name__)
face_path = sys.argv[2]
if not os.path.exists(face_path):
    print(f"Face {face_path} not found")
    sys.exit(1)

# define a video capture object
camera = cv2.VideoCapture(int(sys.argv[1]))

swapper = FaceSwap(upscale=False, profile=True)
source_face = swapper.get_face(sys.argv[2])


def next_frame():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            swapper.swap_face(source_face=source_face, frame=frame)
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # concat frame one by one and show result
    # After the loop release the cap object
    camera.release()


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(next_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
