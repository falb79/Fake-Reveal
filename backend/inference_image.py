from transformers import pipeline

# load deepfake image detection model
detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

"""
function to use deepfake image detection model on input image
returns the label(Real/Fake), and the confidence score
"""
def predict_image(image_path):
    # store the result of the model
    results = detector(image_path)
    # divide into label and score
    label = results[0]['label']
    score = results[0]['score']
    # return the results
    return label, score
