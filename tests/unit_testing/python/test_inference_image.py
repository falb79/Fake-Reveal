# import the required libraries for unit testing
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import numpy as np
import cv2

# add the path of backend model
backend_dir = os.path.abspath(os.path.join(os.getcwd(), "backend"))
sys.path.append(backend_dir)
from inference_image import predict_image 

class TestInferenceImage(unittest.TestCase):

    # setup function to create a temporary image file for testing
    def setUp(self):
        print("\n---Running inference_image tests---")
        
        # create a temporary image array of size 100x100
        temp_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # create a temporary file to save the image 
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp.name, temp_img)
        
        # close the file handler
        temp.close() 
        # save the path of the image in a variable to be used in the test
        self.img_path = temp.name

    # tearDown function to remove the file after finishing the test
    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    
    @patch("inference_image.pipeline")
    def test_predict_image(self, mock_pipeline):
        """
        test case: test the predict_image() returns the label and score successfully

        Docstring for test_predict_image
        
        :param self: instance of the class
        :param mock_pipeline: mock object of pipeline function
        """
        
        # create a mock model and configure its return data
        mock_model = MagicMock()
        mock_model.return_value = [
            {"label": "Real", "score": 0.6}
        ]
        
        # make the mock pipeline return our model
        mock_pipeline.return_value = mock_model
        
        # test the function on the temporary image
        label, score = predict_image(self.img_path)
        
        # check the label and score 
        self.assertEqual(label, "Real")
        self.assertAlmostEqual(score, 0.6, places=2)
        

if __name__ == "__main__":
    unittest.main()