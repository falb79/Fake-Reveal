import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys, os
# add the path of the backend folder
backend_dir = os.path.abspath(os.getcwd()+"/backend")
sys.path.append(backend_dir)
from preprocessing import detect_landmark, preprocess_video

# target for mocking cv2.cvtColor
CVTCOLOR = 'preprocessing.cv2.cvtColor'

class TestPreprocessing(unittest.TestCase):
    # initial setup 
    def setUp(self):
        print("\n---Running preprocessing tests---")
        # create a mock image array, and gray scale image array
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.mock_gray_image = np.zeros((100, 100), dtype=np.uint8)
        
        # mock coordinates for the first 68 landmarks
        # it will be used to verify the final output structure
        self.expected_coords = np.array([
            (10, 20), (11, 21), (12, 22), (13, 23), 
        ] + [(i, i + 10) for i in range(4, 68)], dtype=np.int32)
        
        # create a mock object of the predictor's output 
        self.mock_shape = MagicMock()
        
        # configure the x,y coordinates for all 68 landmarks
        for i in range(68):
            mock_part = MagicMock()
            mock_part.x = self.expected_coords[i, 0]
            mock_part.y = self.expected_coords[i, 1]
            # the predictor's output part function returns the coordinates
            self.mock_shape.part.return_value = mock_part
            # used to iterate through the entire landmark extraction loop, 
            # providing 68 unique coordinate pairs
            self.mock_shape.part.side_effect = lambda index: MagicMock(
                x=self.expected_coords[index, 0], 
                y=self.expected_coords[index, 1]
            )

        # create a mock of the detector and predictor
        self.mock_detector = MagicMock()
        self.mock_predictor = MagicMock(return_value=self.mock_shape)


    @patch(CVTCOLOR)
    def test_detect_landmark_success(self, mock_cvtColor):
        """
        test case: test successful detection of landmarks

        Docstring for test_detect_landmark_success
        
        :param self: instance of the class 
        :param mock_cvtColor: mock object of cv2.cvtColor
        """
        # the return value of cv2.cvtColor is the gray image array
        mock_cvtColor.return_value = self.mock_gray_image
        # mock dlib.rectangle
        self.mock_detector.return_value = ['mock_rect'] 

        # run the detect_landmark() function
        result = detect_landmark(self.test_image, self.mock_detector, self.mock_predictor)

        # check that cvtcolor is called 
        mock_cvtColor.assert_called_once()
        
        # check that the detector and predictor are called
        self.mock_detector.assert_called_once_with(self.mock_gray_image, 1)
        self.mock_predictor.assert_called_once_with(self.mock_gray_image, 'mock_rect')
        
        # check the result
        self.assertIsInstance(result, np.ndarray, "The result should be a NumPy array.")
        self.assertEqual(result.shape, (68, 2), "The array is of shape (68, 2).")
        self.assertTrue(np.array_equal(result, self.expected_coords), 
                        "The result should match the mocked expected coordinates.")


    @patch(CVTCOLOR)
    def test_detect_landmark_no_face_detected(self, mock_cvtColor):
        """
        test case: test detect_landmark() when no face is detected

        Docstring for test_detect_landmark_no_face_detected
        
        :param self: instance of the class 
        :param mock_cvtColor: mock object of cv2.cvtColor
        """""
        # the return value of cv2.cvtColor is the gray image array
        mock_cvtColor.return_value = self.mock_gray_image
        # return an empty list of rectangles
        self.mock_detector.return_value = [] 

        # run the detect_landmark() function
        result = detect_landmark(self.test_image, self.mock_detector, self.mock_predictor)

        # assert
        self.assertIsNone(result, 
                          "The function should return None if no faces are detected.")
        # check that the detector is called but the predictor is not (since no face was detected)
        self.mock_detector.assert_called_once()
        self.mock_predictor.assert_not_called()


    @patch(CVTCOLOR)
    def test_detect_landmark_multiple_faces_detected(self, mock_cvtColor):
        """
        test case: test detect_landmark() handles when multiple faces are detected
        Docstring for test_detect_landmark_multiple_faces_detected
        
        :param self: instance of the class
        :param mock_cvtColor: mock object of cv2.cvtColor
        """
        # the return value of cv2.cvtColor is the gray image array
        mock_cvtColor.return_value = self.mock_gray_image
        
        # define distinct coordinates for the second face
        expected_coords_last_face = np.array([
            (100, 200), (101, 201), (102, 202), 
        ] + [(i + 100, i + 200) for i in range(3, 68)], dtype=np.int32)
        
        # set up a second mock shape for the second face
        mock_shape_2 = MagicMock()
        for i in range(68):
            # used to iterate through the entire landmark extraction loop, 
            # providing 68 unique coordinate pairs
            mock_shape_2.part.side_effect = lambda index: MagicMock(
                x=expected_coords_last_face[index, 0], 
                y=expected_coords_last_face[index, 1]
            )
            
        # configure the predictor to return different shapes for different calls
        self.mock_detector.return_value = ['rect_1', 'rect_2'] 
        
        # execute side_effect callable every time the function is called 
        self.mock_predictor.side_effect = [
            self.mock_shape,      # returned on 1st call (for rect_1)
            mock_shape_2       # returned on 2nd call (for rect_2)
        ]

        #  run the detect_landmark() function
        result = detect_landmark(self.test_image, self.mock_detector, self.mock_predictor)

        # --- Assert ---
        # verify predictor calls
        self.assertEqual(self.mock_predictor.call_count, 2, 
                         "Predictor should be called twice (for each detected face, in this case 2)")
        self.mock_predictor.assert_any_call(self.mock_gray_image, 'rect_1')
        self.mock_predictor.assert_called_with(self.mock_gray_image, 'rect_2')
        
        # check output
        # the result must match the landmarks from the LAST face (mock_shape_2)
        self.assertTrue(np.array_equal(result, expected_coords_last_face), 
                        "The result should contain the landmarks of the last detected face.")


    @patch(CVTCOLOR)
    def test_detect_landmark_detector_exception(self, mock_cvtColor):
        """
        test case: test detect_landmark() when detector raises an exception

        Docstring for test_detect_landmark_detector_exception
        
        :param self: instance of the class
        :param mock_cvtColor: mock object of cv2.cvtColor
        """
        # Arrange
        # the return value of cv2.cvtColor is the gray image array
        mock_cvtColor.return_value = self.mock_gray_image
        # return an error when detector fails
        self.mock_detector.side_effect = RuntimeError("Detector failed to process image.")

        # Act
        # check that it returns an error
        with self.assertRaises(RuntimeError):
            detect_landmark(self.test_image, self.mock_detector, self.mock_predictor)

        # check that the detector is called but the predictor is not
        self.mock_detector.assert_called_once()
        self.mock_predictor.assert_not_called()


    @patch(CVTCOLOR)
    def test_detect_landmark_predictor_exception(self, mock_cvtColor):
        """
        test case: test detect_landmark() when predictor raises an exception
        Docstring for test_detect_landmark_predictor_exception
        
        :param self: instance of the class
        :param mock_cvtColor: mock object of cv2.cvtColor
        """
        # Arrange
        # the return value of cv2.cvtColor is the gray image array
        mock_cvtColor.return_value = self.mock_gray_image
        # mock dlib.rectangle
        self.mock_detector.return_value = ['mock_rect_1']
        # return an error when predictor fails
        self.mock_predictor.side_effect = RuntimeError("Predictor could not find shape.")

        # Act
        # check that it returns an error
        with self.assertRaises(RuntimeError):
            detect_landmark(self.test_image, self.mock_detector, self.mock_predictor)

        # check that both the detector and predictor are called
        self.mock_detector.assert_called_once()
        self.mock_predictor.assert_called_once()


    @patch("preprocessing.write_video_ffmpeg")
    @patch("preprocessing.crop_patch")
    @patch("preprocessing.landmarks_interpolate")
    @patch("preprocessing.detect_landmark")
    @patch("preprocessing.skvideo.io.vread")
    @patch("preprocessing.dlib.shape_predictor")
    @patch("preprocessing.dlib.get_frontal_face_detector")
    @patch("preprocessing.np.load")
    def test_preprocess_video_success(
        self, mock_npload, mock_detector, mock_predictor,
        mock_vread, mock_detect, mock_interpolate,
        mock_crop, mock_write
    ):
        """
        test case: test that preprocess_video() runs seuccessfully

        Docstring for test_preprocess_video_success
        
        :param self: instance of the class
        :param mock_npload: mock object of np.load
        :param mock_detector: mock object of dlib detector
        :param mock_predictor: mock object of dlib predictor
        :param mock_vread: mock object of skvideo.io.vread
        :param mock_detect: mock object of detect_landmark()
        :param mock_interpolate: mock object of landmarks_interpolate()
        :param mock_crop: mock object of crop_patch
        :param mock_write: mock object of write_video_ffmpeg
        """
        # return the mean_face_landmarks model
        mock_npload.return_value = "mean_face_landmarks"

        # using a mock dlib detector and predictor
        mock_detector.return_value = "test_detector"
        mock_predictor.return_value = "test_predictor"

        # use a mock video with two frames
        mock_vread.return_value = np.array([
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8)
        ])

        # return a mock array of landmarks
        mock_detect.return_value = np.zeros((68, 2), dtype=np.int32)

        # mock the behavior of landmarks_interpolate()
        mock_interpolate.return_value = "smooth_landmarks"

        # mock the path to the cropped video
        mock_crop.return_value = "cropped_roi"

        preprocess_video(
            "input.mp4",
            "output.mp4",
            "shape_predictor.dat",
            "mean_face_landmarks.npy"
        )

        # assert
        # check that the detector is called once per frame
        self.assertEqual(mock_detect.call_count, 2)  
        mock_interpolate.assert_called_once()
        mock_crop.assert_called_once()
        mock_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
