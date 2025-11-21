from transformers import pipeline

# Deepfake image detection
detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

def predict_image(image_path):
    """
    ترجع نتيجة: Real/Fake ونسبة الثقة
    """
    results = detector(image_path)
    label = results[0]['label']
    score = results[0]['score']
    return label, score
