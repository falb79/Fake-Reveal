# import the required libraries for unit testing 
import unittest
from unittest.mock import patch, MagicMock
import sys, os
import torch
# add the path of the backend folder
backend_dir = os.path.abspath(os.getcwd()+"/backend")
sys.path.append(backend_dir)
from inference import predict, get_text_from_lip_reading, get_text_from_stt, classify_input

# create a test class that inherits from unittest.TestCase
class TestInferenceVideo(unittest.TestCase):

    # setup function to print a start message for the test
    def setUp(self):
        print("\n---Running inference_video tests---")

    @patch("backend.inference.cv2.VideoCapture")
    @patch("backend.inference.utils")  
    @patch("backend.inference.checkpoint_utils.load_model_ensemble_and_task")
    @patch("backend.inference.tasks.setup_task")
    def test_predict_success(
        self,
        mock_setup_task,
        mock_load_ensemble,
        mock_utils,
        mock_videocap
    ):
        """
        test case: test that predict() works correctly

        Docstring for test_predict_success
        
        :param self: instance of the class
        :param mock_setup_task: mock object of the setup_task function
        :param mock_load_ensemble: mock object of load_model_ensemble_and_task function
        :param mock_utils: mock object of utils functions
        :param mock_videocap: mock object of cv2.VideoCapture
        """

        # mock the return values of the functions to prevent errors
        mock_utils.import_user_module.return_value = True
        mock_utils.split_paths.return_value = []

        # create mock video frames to be returned by cv2.VideoCapture
        mock_cap = MagicMock()
        mock_cap.get.return_value = 10
        mock_videocap.return_value = mock_cap

        # create a mock model to be returned in load_model_ensemble_and_task
        fake_model = MagicMock()
        mock_load_ensemble.return_value = ([fake_model], MagicMock(), MagicMock())
        # create a mock task to be returned in load_model_ensemble_and_task
        fake_task = MagicMock()
        mock_setup_task.return_value = fake_task

        # mock the patch iterator
        fake_iterator = MagicMock()
        fake_iterator.__next__.return_value = {"target": torch.tensor([[1, 2, 3]])}
        fake_task.get_batch_iterator.return_value.next_epoch_itr.return_value = fake_iterator

        # task returns a sample inference result
        fake_task.inference_step.return_value = [[{"tokens": torch.tensor([3, 4, 5])}]]

        # mock the decoder behaviour 
        fake_task.target_dictionary = MagicMock()
        fake_task.datasets = {"test": MagicMock()}
        fake_task.datasets["test"].label_processors = [MagicMock(decode=lambda x, y: "decoded")]
        # test the function
        result = predict("video.mp4", "model-checkpoint.pt", "av_hubert/avhubert")
        # assert
        self.assertEqual(result, "decoded")


    def test_predict_no_frames(self):
        """
        test cases: test that predict() raises an exception when video have 0 frames

        Docstring for test_predict_no_frames
        
        :param self: instance of the class
        """

        with patch("inference.cv2.VideoCapture") as mock_cap:
            # mock the case where no frames are detected
            mock_cap.return_value.get.return_value = 0
            # test it returns an exception
            with self.assertRaises(Exception):
                predict("empty.mp4", "model-checkpoint.pt", "av_hubert/avhubert")

    def test_predict_checkpoint_load_fail(self):
        """
        test case: test predict() raises an exception when model checkpoint loading fails

        Docstring for test_predict_bad_checkpoint
        
        :param self: instance of the class
        """

        with patch("inference.cv2.VideoCapture") as mock_cap, \
             patch("inference.checkpoint_utils.load_model_ensemble_and_task") as mock_load:
            # return the number of frames but raise an error on model load fail
            mock_cap.return_value.get.return_value = 25
            mock_load.side_effect = RuntimeError("Checkpoint load failed")

            with self.assertRaises(RuntimeError):
                predict("video.mp4", "checkpoint-failed-load.pt", "av_hubert/avhubert")
            

    @patch("inference.predict")
    def test_get_text_from_lip_reading(self, mock_predict):
        """
        test case: test that get_text_From_lip_reading() returns 
        the text value from predict()

        Docstring for test_get_text_from_lip_reading
        
        :param self: instance of the class
        :param mock_predict: mock object of the predict function
        """
        # make predict() equal the string 'hello'
        mock_predict.return_value = "hello"
        # run the function and get the result
        result = get_text_from_lip_reading("test.mp4")
        # check that the two strings are equal
        self.assertEqual(result, "hello")


    @patch("inference.predict")
    def test_get_text_from_lip_reading_with_predict_error(self, mock_predict):
        """
        test case: test that get_text_From_lip_reading() returns 
        an error when predict() fails

        Docstring for test_get_text_from_lip_reading_with_predict_error

        :param self: instance of the class
        :param mock_predict: mock object of the predict function
        """
        # make predict() equal the string 'hello'
        mock_predict.side_effect = Exception("An error occured")
        # test the function
        with self.assertRaises(Exception):
            get_text_from_lip_reading("test.mp4")

    
    # @patch replaces interactions with external systems with mock objects
    @patch("inference.whisper")
    def test_get_text_from_stt(self, mock_whisper):
        """
        test case: test that get_text_from_stt() loads the Whisper model,
        runs transcribe function and returns the text result

        Docstring for test_get_text_from_stt

        :param self: instance of the class
        :param mock_whisper: mock object of Whisper 
        """
        # mock the model object and its behavior that is returned by whisper.load_model()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "this is a test"}
        # whisper.load_model should return our model
        mock_whisper.load_model.return_value = mock_model
        # get the result 
        result = get_text_from_stt("test.mp4")
        # check it is equal to a test string
        self.assertEqual(result, "this is a test")

        # ensure whisper.load_model was called once with correct model size
        mock_whisper.load_model.assert_called_once_with("medium")

        # ensure model.transcribe was called with the given file
        mock_model.transcribe.assert_called_once_with("test.mp4")

    
    @patch("inference.whisper")
    def test_model_load_failure(self, mock_whisper):
        """
        test case: test model load error 

        Docstring for test_model_load_failure

        :param self: instance of the class
        :param mock_whisper: mock object of Whisper
        """
        # make load_model raise an exception
        mock_whisper.load_model.side_effect = Exception("Failed to load model 'medium'")
        # test the function
        with self.assertRaises(Exception):
            get_text_from_stt("test.mp4")
        mock_whisper.load_model.assert_called_once_with("medium")
            
 
    
    @patch("inference.whisper")
    def test_transcription_failure(self, mock_whisper):
        """
        test case: test transcribe function failure
        
        Docstring for test_transcription_failure

        :param self: instance of the class
        :param mock_whisper: mock object of Whisper
        """
        # mock the model object and its behavior
        mock_model = MagicMock()
        # make load_model raise an exception
        mock_model.transcribe.side_effect = ValueError("File can't be opened or of unsupported format")
        # whisper.load_model should return our model
        mock_whisper.load_model.return_value = mock_model
        # test the function
        with self.assertRaises(ValueError):
            result = get_text_from_stt("failed_test.mp4")
            


    @patch("inference.whisper")
    def test_invalid_transcription_result(self, mock_whisper):
        """
        test case: test that transcribe function handles invalid output

        Docstring for test_invalid_transcription_result

        :param self: instance of the class
        :param mock_whisper: mock object of Whisper
        """
        # mock the model object and its behavior
        mock_model = MagicMock()
        # make the model return None
        mock_model.transcribe.return_value = 111
        # whisper.load_model should return our model
        mock_whisper.load_model.return_value = mock_model
        # run the function and check it returns an empty string
        with self.assertRaises(TypeError):
            result = get_text_from_stt("test.mp4")
      
    
    @patch("inference.whisper")
    def test_missing_text_key(self, mock_whisper):
        """
        test case: test transcribe function with missing text key in the result object

        Docstring for test_missing_text_key

        :param self: instance of the class
        :param mock_whisper: mock object of Whisper
        """
        # mock the model object and its behavior
        mock_model = MagicMock()
        # make the model return results object with no 'text' key 
        mock_model.transcribe.return_value = {"language": "en"}
        # whisper.load_model should return our model
        mock_whisper.load_model.return_value = mock_model
        # run the function and check the results
        with self.assertRaises(KeyError):
            get_text_from_stt("test.mp4")
        self.assertTrue(mock_model.transcribe.called)

    
    @patch("inference.whisper")
    def test_empty_text(self, mock_whisper):
        """
        test case: test transcribe function with empty string results

        Docstring for test_empty_text
        
        :param self: instance of the class
        :param mock_whisper: mock object of Whisper
        """
        # mock the model object and its behavior
        mock_model = MagicMock()
        # make the model return result object with empty string
        mock_model.transcribe.return_value = {"text": ""}
        # whisper.load_model should return our model
        mock_whisper.load_model.return_value = mock_model
        # run the function and check the results
        result = get_text_from_stt("test.mp4")
        self.assertEqual(result, "")
        self.assertTrue(mock_model.transcribe.called)
    
    def test_classify_input(self):
        """
        test case: test that classify_input() calculates the similarity 
        and assigns the correct label
        """
        # run the function to get the similarity and label 
        similarity_score, label = classify_input("hello", "hello")
        # check the results
        self.assertEqual(similarity_score, 100.0)
        self.assertEqual(label, "Real")

if __name__ == "__main__":
    unittest.main()
