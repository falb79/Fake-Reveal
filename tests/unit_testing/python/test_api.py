# import the required libraries for unit testing
import unittest
import sys, os
import warnings
# add the path of the app folder
app_dir = os.path.abspath(os.getcwd()+"/app")
sys.path.append(app_dir)
from api import app

test_video_path = "tests/test_data/test_video.mp4"
test_image_path = "tests/test_data/test_image.jpg"

# create a test class that inherits from unittest.TestCase
class TestAPI(unittest.TestCase):
    # create the client to run the tests on without running the server
    def setUp(self):
        print("\n---Running API tests---")
        warnings.simplefilter("ignore")
        self.client = app.test_client()


    def test_homepage_load(self):
        """
        test case 1: test that the homepage is loaded correctly
        """
        response = self.client.get("/")
        # check that that the response code is 200 (success)
        self.assertEqual(response.status_code, 200)
        # check that the returned content is html
        self.assertIn(b"<html", response.data)


    def test_video_upload_valid(self):
        """
        test case 2: test predict_video endpoint with valid uploaded video 
        """
        # try to send a request
        try:
            # Use a 'with' statement to automatically close the file
            with open(test_video_path, 'rb') as f:
                response = self.client.post(
                    "/predict_video",
                    data={"video": (f, "test_video.mp4")}, 
                    headers={'Content-Type': 'multipart/form-data'},
                )
                # check that that the response code is 200 (success)
                self.assertEqual(response.status_code, 200)
            
            data = response.get_json()
            # check the data in the response
            self.assertIn("label", data)
            self.assertIn("score", data)
            self.assertIn("lip_reading_text", data)
            self.assertIn("speech_text", data)
        # handle errors
        except FileNotFoundError:
            self.fail(f"File not found. Ensure '{test_video_path}' exists.")
        # remove temporary files
        finally:
            if os.path.exists("uploads/temp_video.mp4"):
                os.remove("uploads/temp_video.mp4")

    
    def test_video_upload_invalid(self):
        """
        test case 3: test that predict_video endpoint would return the correct error code
        on unsupported file type upload
        """
        # try to send a request
        response = self.client.post(
            "/predict_video",
            data={"video" : "This is test data."}, 
            headers={'Content-Type': 'application/json'},
        )
        # it should return error 415 (unsupported media type)
        self.assertEqual(response.status_code, 415)

    
    def test_video_upload_no_file(self):
        """
        test case 4: test predict_video endpoint with no video uploaded returns an error
        """
        response = self.client.post(
            "/predict_video",
            headers={'Content-Type': 'multipart/form-data'},
            )
        # it should return error 400 (bad request)
        self.assertEqual(response.status_code, 400)

    
    def test_image_upload_valid(self):
        """
        test case 5: test predict_image endpoint with valid uploaded image 
        """
        # try to send a request
        try:
            # Use a 'with' statement to automatically close the file
            with open(test_image_path, 'rb') as f:
                response = self.client.post(
                    "/predict_image",
                    data={"image": (f, "test_image.jpg")}, 
                    headers={'Content-Type': 'multipart/form-data'},
                )
                # check that that the response code is 200 (success)
                self.assertEqual(response.status_code, 200)
            
            data = response.get_json()
            # check the data in the response
            self.assertIn("label", data)
            self.assertIn("score", data)
        # handle errors
        except FileNotFoundError:
            self.fail(f"File not found. Ensure '{test_image_path}' exists.")
        # remove temporary files
        finally:
            if os.path.exists("uploads/temp_image.jpg"):
                os.remove("uploads/temp_image.jpg")
        

    
    def test_image_upload_invalid(self):
        """
        test case 6: test that predict_image endpoint would return the correct error code
        on unsupported file type upload
        """
       # try to send a request
        response = self.client.post(
            "/predict_image",
            data={'image': 'This is test data.'}, 
            headers={'Content-Type': 'application/json'},
        )
        # it should return error 415 (unsupported media type)
        self.assertEqual(response.status_code, 415)
    
    
    def test_image_upload_no_file(self):
        """
        test case 7: test predict_image endpoint with no image uploaded returns an error
        """
        response = self.client.post(
            "/predict_image",
            headers={'Content-Type': 'multipart/form-data'},
            )
        # it should return error 400 (bad request)
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()