import os
import requests
import json

# --- Settings ---
# Folder containing all test data (Real and Fake)
DATA_DIR = os.getcwd() + '/tests/test_data' 
# Your video API endpoint (Flask server must be running)
API_URL = 'http://127.0.0.1:5000/predict_video' 

# --- Counters ---
results = {
    'Real': {'Correct': 0, 'Incorrect': 0, 'Errors': 0},
    'Fake': {'Correct': 0, 'Incorrect': 0, 'Errors': 0},
}
total_tests = 0

def send_video_to_api(file_path):
    """
    Sends a single video file to the API and retrieves the predicted label.
    """
    try:
        with open(file_path, 'rb') as f:
            files = {'video': (os.path.basename(file_path), f, 'video/mp4')}
            response = requests.post(API_URL, files=files, timeout=1000) # Increase timeout to ensure model responds
            
            if response.status_code == 200:
                data = response.json()
                return data.get('label', 'UNKNOWN')
            else:
                print(f"\n[ERROR] API responded with {response.status_code}. Details: {response.text}")
                return 'ERROR'
    except Exception as e:
        # If connection fails or timeout occurs
        return 'CRITICAL_ERROR'


def evaluate_directory(directory_path, true_label):
    """
    Evaluates all video files in a given folder.
    """
    global total_tests
    
    video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.mov'))]
    print(f"\n--- Testing {len(video_files)} files as TRUE {true_label} ---")
    
    for filename in video_files:
        file_path = os.path.join(directory_path, filename)
        
        # Send the file and get prediction
        predicted_label = send_video_to_api(file_path)
        
        # Update counters
        if predicted_label == true_label:
            results[true_label]['Correct'] += 1
        elif predicted_label in ('Fake', 'Real'): 
            results[true_label]['Incorrect'] += 1
            print(f"  [MISCLASSIFIED] {filename} classified as {predicted_label}")
        else:
            results[true_label]['Errors'] += 1
            print(f"  [FAIL] Failed to process {filename}")
            
        total_tests += 1


def generate_simple_report():
    """
    Calculates and displays basic evaluation results.
    """
    print("\n" + "="*50)
    print("           SIMPLE MODEL ACCURACY REPORT")
    print("="*50)

    # Calculate overall accuracy
    total_correct = results['Real']['Correct'] + results['Fake']['Correct']
    overall_accuracy = (total_correct / total_tests) * 100 if total_tests else 0
    
    print(f"Total Tests Run: {total_tests}\n")

    # Display results for Real category
    real_correct = results['Real']['Correct']
    real_incorrect = results['Real']['Incorrect']
    real_total = real_correct + real_incorrect
    real_accuracy = (real_correct / real_total) * 100 if real_total else 0

    print(f"--- TRUE REAL Videos ({real_total} total) ---")
    print(f"  Correctly Classified: {real_correct}")
    print(f"  Misclassified as Fake: {real_incorrect}")
    print(f"  Accuracy for REAL: {real_accuracy:.2f}%\n")

    # Display results for Fake category
    fake_correct = results['Fake']['Correct']
    fake_incorrect = results['Fake']['Incorrect']
    fake_total = fake_correct + fake_incorrect
    fake_accuracy = (fake_correct / fake_total) * 100 if fake_total else 0

    print(f"--- TRUE FAKE Videos ({fake_total} total) ---")
    print(f"  Correctly Classified: {fake_correct}")
    print(f"  Misclassified as Real: {fake_incorrect}")
    print(f"  Accuracy for FAKE: {fake_accuracy:.2f}%\n")
    
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("="*50)


if __name__ == "__main__":
    
    print("--- SIMPLE EVALUATION START ---")
    print("Ensure your Flask API is running on http://127.0.0.1:5000.")
    input("Press Enter to start the evaluation...")
    
    # Run evaluation
    evaluate_directory(os.path.join(DATA_DIR, 'real'), 'Real')
   # evaluate_directory(os.path.join(DATA_DIR, 'fake'), 'Fake')
    
    # Display report
    generate_simple_report()
