from flask import Flask, request, render_template
from backend.inference_video import predict_video
from backend.inference_image import predict_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    video_result = None
    video_conf = None
    image_result = None
    image_conf = None

    if request.method == 'POST':
        # تحليل الفيديو
        if 'video' in request.files:
            video_file = request.files['video']
            video_path = os.path.join(UPLOAD_FOLDER, "temp_video.mp4")
            video_file.save(video_path)
            video_result, video_conf = predict_video(video_path)

        # تحليل الصورة
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join(UPLOAD_FOLDER, "temp_image.png")
            image_file.save(image_path)
            image_result, image_conf = predict_image(image_path)

    return render_template(
        'index.html',
        video_result=video_result,
        video_conf=video_conf,
        image_result=image_result,
        image_conf=image_conf
    )

if __name__ == '__main__':
    app.run(debug=True)
