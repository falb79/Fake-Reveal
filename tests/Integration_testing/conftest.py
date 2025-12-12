import pytest
import tempfile
import os
from io import BytesIO
from app.api import create_app 

@pytest.fixture
def app():
    """ Initialize a Flask app object for testing environment."""
    #use the create_app function from app/api.py
    flask_app = create_app()
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    # create a test client for the Flask app
    return app.test_client()

@pytest.fixture
def dummy_image():
    #create a dummy image file in memory to simulate upload
    file_data = BytesIO(b"Dummy image content")
    file_data.name = 'test_image.jpg'
    return file_data

@pytest.fixture
def dummy_video():
    file_data = BytesIO(b"Dummy video content")
    file_data.name = 'test_video.mp4'
    return file_data