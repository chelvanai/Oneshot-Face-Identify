import json
from flask import Flask, request, Response
from video_stream import VideoCamera
import os

app = Flask(__name__)

f = None
url = None

if not os.path.isdir('./videos'):
    os.mkdir('./videos')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        global f, url
        url = None
        f = request.files['video']
        f.save('./videos/' + f.filename)
        return json.dumps(True)


@app.route('/ipcam', methods=['GET', 'POST'])
def ipcam():
    if request.method == 'POST':
        global url, f
        f = None
        url = request.form['url']
        return json.dumps(True)


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            return "video or ipcam not connected"
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    if f is None and url is None:
        print("Video IP cam none")
        return "please select video or ipcam url"

    if f is None:
        video_process = VideoCamera(video=None, url=url)
        return Response(gen(video_process),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        video_process = VideoCamera(video='./videos/' + str(f.filename), url=None)
        return Response(gen(video_process),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
