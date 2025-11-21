import random
import time

def predict_video(video_path):
    """
    دالة وهمية لتجربة المشروع
    ترجع نتيجة عشوائية (Real/Fake) ونسبة ثقة.
    """
    time.sleep(2)  # محاكاة تحليل الفيديو
    result = random.choice(["Real", "Fake"])
    confidence = random.uniform(0.7, 0.99)
    return result, confidence
