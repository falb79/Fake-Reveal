# About the project

We present Fake-Reveal, an AI-powered platform dedicated to detecting and revealing deepfake videos.

It analyzes video content using advanced deep learning models to identify subtle inconsistencies in lip movements and facial features. The primary goal is to provide a reliable tool for verifying video authenticity, helping both researchers and the public combat fake media content.


## Prerequisites:
- Cmake 
   
   Download link:  `https://cmake.org/download/`

   **Choose:** Windows x64 Installer or macOS
   
   **Note:** Enable *Add to PATH* during installation.  

- ffmpeg

   Download link: `https://www.gyan.dev/ffmpeg/builds/`
   
   **Choose:** `ffmpeg-git-full.7z`
   
   **Note:** Unzip the folder, then use PowerShell (Run as Administrator) to add FFmpeg to SYSTEM PATH

   
- Conda
   
   Install Anaconda or Miniconda.

- C++ Build Tools

   Download link: `https://visualstudio.microsoft.com/visual-cpp-build-tools/`
   
   **Note:** Choose C++ build tools when installing

   
## Project setup for the first time: 

### 1. Clone our project's repository 
`git clone https://github.com/falb79/Fake-Reveal`

### 2. Clone AV-HuBERT repository (inside the project's folder)
In your terminal run: 

```
git clone https://github.com/facebookresearch/av_hubert.git

cd av_hubert

git submodule init

git submodule update
```
### 3. Create a virtual environment using conda  
In a new terminal run:
```
conda create -n venv_py39 python=3.9

conda activate venv_py39

conda install pip<24.1
```
### 4. Install requirements
change to the project directory (cd path/to/project), and run:

`pip install -r requirements.txt` 

now make sure you're in the project/av_hubert directory (cd av_hubert), then run:
```
cd fairseq

pip install --editable ./
```

**Note**: If you face permission issues, run Anaconda CMD as Administrator.


### 5. Download checkpoint and files needed for preprocessing

Create a folder inside the project named (models) and add:

   - Fine-tuned model

   - shape_predictor_68_face_landmarks.dat

   - 20words_mean_face.npy

These files can be found in the AV-HuBERT repository `https://github.com/facebookresearch/av_hubert` or the demo files provided.

### 6. Create required additional folders

📂Inside the project directory, create these empty folders:

   - uploads
   
   - mouth_roi



## To Run The Project:
Open the project in VS Code, then in the terminal:
```
conda activate venv_py39

python app/api.py
```

**Note:** to close the virtual environment run: conda deactivate 

## To Run Python Tests:

### First create required test data folder

📂Inside the tests folder, create a test_data folder containing:
   - two files: test_video.mp4 and test_image.jpg (for API unit test)
   - real folder: this contains the real videos from the dataset (for model testing)
   - fake folder: this contains the fake videos from the dataset (for model testing)
   

Open the project in VS Code, then in the terminal:
`conda activate venv_py39`

### To run model test:
run: `python tests/model_testing/evaluate.py`

### To run unit test:

**To run a specific test file use: python -m unittest tests.unit_testing.python.fileNameHere**
For example to run `test_api.py`: `python -m unittest tests.unit_testing.python.test_api`

**To run a specific test cases use: python -m unittest tests.unit_testing.python.fileNameHere.className.functionName**
For example to run `test_homepage_load` from `test_api.py`: `python -m unittest tests.unit_testing.python.test_api.TestAPI.test_homepage_load`


## Citations
We acknowledge and are grateful for the foundational work provided by the following open-source projects and research papers.

1. [AV-HuBERT lip reading model](https://github.com/facebookresearch/av_hubert)


2. [Deepfake vs real faces detection ViT](https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit/notebook)


3. [OpenAI whisper](https://openai.com/index/whisper/)
