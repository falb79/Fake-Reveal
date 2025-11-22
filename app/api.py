import os, sys
from flask import Flask, request, jsonify, render_template
backend_dir = os.path.abspath("backend")
sys.path.append(backend_dir)
from inference import get_text_from_lip_reading, get_text_from_stt, classify_input
from preprocessing import preprocess_video
from inference_image import predict_image

face_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
mean_face_path = "models/20words_mean_face.npy"
mouth_roi_path = "mouth_roi/roi.mp4"

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]

    # Save temporary video file
    video_path = "uploads/temp_video.mp4"
    video_file.save(video_path)
    try:
        # Run lip reading
        preprocess_video(video_path, mouth_roi_path, face_predictor_path, mean_face_path)
        lip_text = get_text_from_lip_reading(mouth_roi_path)

        # Run STT on audio
        audio_text = get_text_from_stt(video_path)

        # Compare similarity
        similarity = classify_input(lip_text, audio_text)

        return jsonify({
            "lip_reading_text": lip_text,
            "speech_text": audio_text,
            "classification": similarity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route('/predict_image', methods=['POST'])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files["image"]

    # Save temporary video file
    image_path = "uploads/temp_image.jpg"
    image_file.save(image_path)

    label, score = predict_image(image_path)
    return jsonify({
            "label": label,
            "score": score
        })
    


@app.route("/", methods=["GET"])
def root():
    #return "API running!"
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
