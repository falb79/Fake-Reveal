# import the required libraries 
import os, sys
from flask import Flask, request, jsonify, render_template
# add the path to the backend models
backend_dir = os.path.abspath("backend")
sys.path.append(backend_dir)
from inference import get_text_from_lip_reading, get_text_from_stt, classify_input
from preprocessing import preprocess_video
from inference_image import predict_image

# define the paths to the tools for preprocessing 
face_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
mean_face_path = "models/20words_mean_face.npy"
# define the path to save the mouth roi video after preprocessing
mouth_roi_path = "mouth_roi/roi.mp4"

# create a Flask app
app = Flask(__name__)

# define the route for using video deepfake detection model
@app.route('/predict_video', methods=['POST'])
def predict_video():
    # check for unsupported file types and return an error
    if request.mimetype != 'multipart/form-data':
        return jsonify({"error": "Unsupported Media Type"}), 415
    # check if no video was uploaded and return an error
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    # get the video file from the request 
    video_file = request.files["video"]

    # save temporary video file
    video_path = "uploads/temp_video.mp4"
    video_file.save(video_path)
    try:
        # preprocess the video and get text from lip reading model
        preprocess_video(video_path, mouth_roi_path, face_predictor_path, mean_face_path)
        lip_text = get_text_from_lip_reading(mouth_roi_path)

        # get text from speech-to-text model
        audio_text = get_text_from_stt(video_path)

        # compare similarity
        similarity, label = classify_input(lip_text, audio_text)

        # format the result as a JSON object
        result = jsonify({
            "label": label,
            "score": f"{similarity:.2f}",
            "lip_reading_text": lip_text,
            "speech_text": audio_text,
        })
        # add header to the response
        result.headers.add("Access-Control-Allow-Origin", "*")
        # return the result 
        return result
    # handle exceptions
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # remove the temporary video file
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/predict_image', methods=['POST'])
def predict_image_output():
    # check for unsupported file types and return an error
    if request.mimetype != 'multipart/form-data':
        return jsonify({"error": "Unsupported Media Type, please use multipart/form-data"}), 415
        
    # check if no image was uploaded and return an error
    # Note: request.files is usually empty if mimetype is wrong, but this check remains valid
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Initialize path to None for safe cleanup in case of early error
    image_path = None
    try:
        # get the image file from the request 
        image_file = request.files["image"]

        # save temporary image file
        image_path = "uploads/temp_image.jpg"
        image_file.save(image_path)

        # get the results from the image deepfake detection model
        label, score = predict_image(image_path)

        # format the result as a JSON object
        result = jsonify({
                    "label": label,
                    "score": f"{score*100:.2f}"
                })
        # add header to the response
        result.headers.add("Access-Control-Allow-Origin", "*")

        # return the result
        return result

    except Exception as e:
        # Handle internal errors (e.g., model crash, file save error)
        print(f"Internal error during image prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # remove temporary image file even if an error occurred (Cleanup guarantee)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
                

# define the main route to the website
@app.route("/", methods=["GET"])
def root():
    return render_template('index.html')

# run the app
if __name__ == "__main__":
    app.run(debug=True)

def create_app():
    return app