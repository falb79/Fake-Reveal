from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import requests
import cv2, os, numpy as np
import subprocess, glob
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
COLAB_API_URL = "https://surpass-dominoes-squatting.ngrok-free.dev/process"

@app.route('/')
def index():
    return render_template('index.html')

session = requests.Session()
session.trust_env = False 


@app.route('/send_to_colab', methods=['POST'])
def send_to_colab():
    try:
        # clear uploads folder at the start of each process
        files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error cleaning up old file {f}: {e}")
        # get the video from user request
        video_file = request.files['video']
        # get the filename of the video
        filename = video_file.filename
        # store the file for different stages of the processing
        # 1. raw_path: initial input video 
        raw_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), f"raw_{filename}")
        # 2. render_path: the output of OpenCV processing (no audio)
        render_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), f"render_{filename}")
        # 3. final_path: the final processed video with overlay (with audio, browser compatible)
        final_filename = f"final_{filename}"
        final_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), final_filename)
        video_file.save(raw_path)

        # send to Colab for segment analysis
        print(f"Sending {filename} to Colab...")
        with open(raw_path, 'rb') as f:
            files = {'video': (filename, f, video_file.content_type)}
            response = session.post(COLAB_API_URL, files=files, timeout=10000)

        if response.status_code != 200:
            return jsonify({"error": "Colab analysis failed"}), 500
        
        # get the segments from the response
        segments = response.json().get('segments', [])
        print(f"Received {len(segments)} segments from Colab.")

        # OpenCV Rendering
        cap = cv2.VideoCapture(raw_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # using mp4v for initial render
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(render_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # apply highlights based on Colab segments
            for seg in segments:
                start, end = seg.get('start'), seg.get('end')
                if start is not None and end is not None:
                    if start <= curr_time <= end:
                        color = (0, 0, 255) # BGR
                        overlay = np.full(frame.shape, color, dtype=np.uint8)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        break
            out.write(frame)

        cap.release()
        out.release()

        # convert to FFmpeg for HTML5 compatibility
        print("Converting to H.264...")
        cmd = (
            f'ffmpeg -i "{render_path}" -i "{raw_path}" '
            f'-map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p '
            f'-c:a aac -movflags +faststart -shortest -y "{final_path}"'
        )
        subprocess.run(cmd, shell=True)

        # return JSON of the final results
        res = jsonify({
            "status": "success",
            "video_url": f"/get_video/{final_filename}",
            "segments": segments,
            "prediction": response.json().get('prediction'),
            "confidence_score": response.json().get('confidence_score'),
            "whisper_text": response.json().get('whisper_text'),
            "avhubert_text": response.json().get('avhubert_text')
        })
        res.headers['Access-Control-Allow-Origin'] = '*'
        return res

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_video/<filename>')
def get_video(filename):
    filename = secure_filename(filename)
    # log what the browser is requesting 
    print(f"Browser is asking for: {filename}") 
    
    # check if the file exists before sending
    file_path = os.path.join(os.path.abspath(UPLOAD_FOLDER), filename)
    if not os.path.exists(file_path):
        print(f"ERROR: {os.path.relpath(file_path)} does not exist!")
        return "File not found", 404

    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

if __name__ == '__main__':
    app.run(port=8000, debug=True)