# import libraries for unit test
import unittest
import io
import os
from unittest.mock import patch, MagicMock
from app.app import app, UPLOAD_FOLDER

# create a test class that inherits from unittest.TestCase
class FlaskAPITest(unittest.TestCase):
    # setup function to print a start message for the test
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        print(f"\nRUNNING: {self._testMethodName}...", end=" ")

    def tearDown(self):
        # clean up any test files created during a specific test
        pass

    # --- Test Case 1 ---
    def test_index_route_endpoint(self):
        """Test that the HTML template renders correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # check for a keyword from the UI
        self.assertIn(b'Fake Reveal', response.data)
        print("✅ SUCCESS: HTML Template rendered.")

    # --- Test Case 2 ---
    # patch dependencies
    @patch('app.app.requests.Session.post')
    @patch('app.app.subprocess.run')
    def test_process_video_success(self, mock_ffmpeg, mock_colab):
        """Test successful video processing"""
        
        # mock connection to Colab API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'segments': [{'start': 1.0, 'end': 2.0}]
        }
        mock_colab.return_value = mock_response

        # mock FFmpeg
        mock_ffmpeg.return_value = MagicMock(returncode=0)
        # create the data dictionary
        data = {
            'video': (io.BytesIO(b"api test video data"), 'test_video.mp4')
        }
        # test the request
        response = self.client.post('/send_to_colab', 
                                    data=data, 
                                    content_type='multipart/form-data')
        # assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'success')
        print("✅ SUCCESS: Video processed and endpoints reached.")

    # --- Test Case 3 ---
    @patch('app.app.send_from_directory')
    def test_get_video_success(self, mock_send):
        """Test that get_video successfully points to the directory."""
        from flask import Response
        
        # use absolute path for comparison
        absolute_upload_folder = os.path.abspath(UPLOAD_FOLDER)
        
        # mock the response 
        mock_send.return_value = Response("unit test content", status=200, mimetype='video/mp4')
        # mock test file
        test_file = "final_test.mp4"
        test_path = os.path.join(absolute_upload_folder, test_file)

        if not os.path.exists(absolute_upload_folder):
            os.makedirs(absolute_upload_folder)
        with open(test_path, 'w') as f:
            f.write("unit test content")

        try:
            response = self.client.get(f'/get_video/{test_file}')

            # mock_send.call_args to see exactly what was passed
            args, kwargs = mock_send.call_args
            actual_path = args[0]
            
            # compare absolute paths (what was passed vs what was expected)
            self.assertEqual(os.path.abspath(actual_path), absolute_upload_folder)
            self.assertEqual(args[1], test_file)
            
            self.assertEqual(response.status_code, 200)
            print(f"✅ SUCCESS: get_video correctly matched path: {os.path.relpath(actual_path)}")

        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    # --- Test Case 4 ---
    def test_get_video_not_found(self):
        """Test passing a video that doesn't exist."""
        response = self.client.get('/get_video/non_existent.mp4')
        self.assertEqual(response.status_code, 404)
        print("✅ SUCCESS: 404 correctly handled.")

if __name__ == '__main__':
    unittest.main()