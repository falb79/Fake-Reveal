import pytest
import json
from unittest.mock import patch, MagicMock
from io import BytesIO
#dummy_image and client are imported from conftest.py automatically


# --- SUCCESS CASES ---

# 1- Real image test 
#import paths for patching should match those in app/api.py
@patch('app.api.predict_image') 
@patch('os.remove') #clean up temporary file
def test_predict_image_success_Real(mock_remove, mock_predict_image, client, dummy_image):
    """ Test the /predict_image route for successful response."""

    #mock the predict_image function to return a real label and score
    mock_predict_image.return_value = ("Real", 0.998)
    
    # send the request to the endpoint 
    response = client.post(
        '/predict_image',
        data={'image': (dummy_image, 'test_image.jpg')},
        content_type='multipart/form-data'
    )
    
    #verify the response
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['label'] == "Real"
    assert data['score'] == "99.80"  
    
    # ensure the mocked functions were called
    mock_predict_image.assert_called_once()
    mock_remove.assert_called_once() # ensure temporary file is removed

# 2-Fake image test
@patch('app.api.predict_image') 
@patch('os.remove')
def test_predict_image_success_fake(mock_remove, mock_predict_image, client, dummy_image):
    """ Test the /predict_image route for successful Fake response """

    # Mock the predict_image function to return a fake label and score
    mock_predict_image.return_value = ("Fake", 0.950) 
    
    # Send the request
    response = client.post(
        '/predict_image',
        data={'image': (dummy_image, 'test_image.jpg')},
        content_type='multipart/form-data'
    )
    
    # Assertions
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['label'] == "Fake"
    assert data['score'] == "95.00" 
    
    # Ensure the mocked functions were called
    mock_predict_image.assert_called_once()
    mock_remove.assert_called_once()

#--- FAILURE CASES ---

@patch('app.api.predict_image')
@patch('os.remove')
def test_predict_image_failure_internal(mock_remove, mock_predict_image, client, dummy_image):
    """
    Test the /predict_image route for internal failure (500).
    Simulates a crash during model execution or file handling after upload.
    """
    
    # mock function to raise an exception
    mock_predict_image.side_effect = Exception("Model initialization failed.")
    
    # Send the request
    response = client.post(
        '/predict_image',
        data={'image': (dummy_image, 'test_image.jpg')},
        content_type='multipart/form-data'
    )
    
    # Assertions
    assert response.status_code == 500
    data = json.loads(response.data)
    
    # Ensure the error message matches the raised exception
    assert 'error' in data
    assert "Model initialization failed." in data['error'] 
    
    # Ensure cleanup was attempted even in case of failure
    mock_remove.assert_called_once()



def test_predict_image_no_file(client):
    """Test the /predict_image route for error when no file is uploaded."""
    
    response = client.post(
        '/predict_image',
        data={}, # no file uploaded
        content_type='multipart/form-data'
    ) 
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] == "No image file uploaded"