import pytest
import json
from unittest.mock import patch, MagicMock

#Test Case 1: a successful real video case 
#to ensure the patch paths match those in app/api.py
@patch('app.api.get_text_from_stt')
@patch('app.api.get_text_from_lip_reading')
@patch('app.api.preprocess_video')
@patch('os.remove') 
def test_predict_video_real_case(mock_remove,mock_preprocess, mock_lip_reading, mock_stt, client,  dummy_video):
    """ test the /predict_video route for 'Real' case (matching texts)."""

    #mocking the functions to return matching texts
    mock_lip_reading.return_value = "THE CAT IS BLUE" 
    mock_stt.return_value = "THE CAT IS BLUE"
    mock_preprocess.return_value = None 
    
    #sending the request
    response = client.post(
        '/predict_video',
        data={'video': (dummy_video, 'test_video.mp4')},
        content_type='multipart/form-data'
    )
    
    # verifying the response
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['label'] == "Real"
    #both texts are identical, so similarity should be 100%
    assert data['score'] == "100.00" 
    assert data['lip_reading_text'] == "THE CAT IS BLUE"
    assert data['speech_text'] == "THE CAT IS BLUE"

    #make sure all functions were called
    mock_preprocess.assert_called_once()
    mock_lip_reading.assert_called_once()
    mock_stt.assert_called_once()
    mock_remove.assert_called_once()


#Test Case 2: a successful fake video case 
@patch('app.api.get_text_from_stt')
@patch('app.api.get_text_from_lip_reading')
@patch('app.api.preprocess_video')
@patch('os.remove')
def test_predict_video_fake_case(mock_remove,mock_preprocess, mock_lip_reading, mock_stt, client, dummy_video):
    """test the /predict_video route for 'Fake' case (non-matching texts).""" 

    #mocking the functions to return different texts
    mock_lip_reading.return_value = "I love Saudi Arabia" 
    mock_stt.return_value = "THE DOG IS RED" 
    mock_preprocess.return_value = None 
    
    #send the request
    response = client.post(
        '/predict_video',
        data={'video': (dummy_video, 'test_video.mp4')},
        content_type='multipart/form-data'
    )
    
    # verify the response
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # it should be classified as Fake
    assert data['label'] == "Fake" 
    
    # note the (Fake: similarity = 100 - similarity) ,so it should be high
    score_float = float(data['score'])
    assert score_float > 80.0
    
    
    mock_lip_reading.assert_called_once()
    mock_stt.assert_called_once()


# Test Case 3: Preprocessing Failure
@patch('app.api.get_text_from_stt')
@patch('app.api.get_text_from_lip_reading')
@patch('app.api.preprocess_video')
@patch('os.remove')
def test_predict_video_failure_preprocess(mock_remove, mock_preprocess, mock_lip_reading, mock_stt, client, dummy_video):
    """Test the /predict_video route fails (500) when preprocessing raises an exception."""
    
    # Setup the mock to raise an exception during preprocessing
    mock_preprocess.side_effect = Exception("Video processing failed: Face not detected.")
        
    # Send the request
    response = client.post(
        '/predict_video',
        data={'video': (dummy_video, 'test_video.mp4')},
        content_type='multipart/form-data'
    )
    
    # Assertions
    assert response.status_code == 500
    data = json.loads(response.data)
    
    assert 'error' in data
    assert "Video processing failed: Face not detected." in data['error']
    
    mock_remove.assert_called_once()
    mock_lip_reading.assert_not_called()
    mock_stt.assert_not_called()


# Test Case 4: Lip Reading Model Failure
@patch('app.api.get_text_from_stt')
@patch('app.api.get_text_from_lip_reading')
@patch('app.api.preprocess_video')
@patch('os.remove')
def test_predict_video_failure_lip_reading(mock_remove, mock_preprocess, mock_lip_reading, mock_stt, client, dummy_video):
    """Test the /predict_video route fails (500) when the lip reading model crashes."""
    
    # Setup the mocks
    mock_preprocess.return_value = None # Preprocessing succeeds
    mock_lip_reading.side_effect = Exception("Lip model out of memory.") # Lip reading fails
    
    # Send the request
    response = client.post(
        '/predict_video',
        data={'video': (dummy_video, 'test_video.mp4')},
        content_type='multipart/form-data'
    )
    
    # Assertions
    assert response.status_code == 500
    data = json.loads(response.data)
    
    assert 'error' in data
    assert "Lip model out of memory." in data['error']
    
    # Verify that the next function (stt) was NOT called, but cleanup was attempted
    mock_preprocess.assert_called_once()
    mock_lip_reading.assert_called_once()
    mock_stt.assert_not_called() 
    mock_remove.assert_called_once()

# Test Case 5: STT Model Failure
@patch('app.api.get_text_from_stt')
@patch('app.api.get_text_from_lip_reading')
@patch('app.api.preprocess_video')
@patch('os.remove')
def test_predict_video_failure_stt(mock_remove, mock_preprocess, mock_lip_reading, mock_stt, client, dummy_video):
    """Test the /predict_video route fails (500) when the STT model crashes."""
    
    # 1. Setup the mocks
    mock_preprocess.return_value = None # Preprocessing succeeds
    mock_lip_reading.return_value = "Success reading lips" # Lip reading succeeds
    mock_stt.side_effect = Exception("STT model failed to extract audio.") # STT fails
    
    # 2. Send the request
    response = client.post(
        '/predict_video',
        data={'video': (dummy_video, 'test_video.mp4')},
        content_type='multipart/form-data'
    )
    
    # 3. Assertions
    assert response.status_code == 500
    data = json.loads(response.data)
    
    assert 'error' in data
    assert "STT model failed to extract audio." in data['error']
    
    # Verify the sequence up to the crash point
    mock_preprocess.assert_called_once()
    mock_lip_reading.assert_called_once()
    mock_stt.assert_called_once()
    mock_remove.assert_called_once()


# Test Case 6: No Video Uploaded
def test_predict_video_no_file(client):
    """test the /predict_video route when no video file is uploaded."""

    response = client.post(
        '/predict_video',
        data={}, # no file uploaded
        content_type='multipart/form-data'
    ) 

    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == "No video file uploaded"