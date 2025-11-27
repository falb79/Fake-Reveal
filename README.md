## Prerequisites:
1. Download cmake - required for dlib

   https://cmake.org/download/

   choose (Windows x64 Installer) or (macOS) 
2. Download ffmpeg

   https://www.gyan.dev/ffmpeg/builds/

   choose (ffmpeg-git-full.7z)

3. Add both cmake and ffmpeg to system path

   for cmake, add it to system path while installation
   
   for FFmpeg, unzip the folder, then use PowerShell (Run as Administrator) to add FFmpeg to SYSTEM PATH
   
4. Install conda (if not already installed)

## To use the project for the first time: 

### 1. Clone our project's repository 
`git clone https://github.com/falb79/DeepfakeVideoDetection.git`

### 2. Clone AV-HuBERT repository inside the project's folder
In your terminal run: 

```
git clone https://github.com/facebookresearch/av_hubert.git

cd av_hubert

git submodule init

git submodule update
```
### 3. Create a virtual environment using conda, and install requirements  
In a new terminal run:
```
conda create -n venv_py39 python-3.9

conda activate venv_py39

conda install pip<24.1
```
change to the project directory (cd path/to/project), and run:

`pip install -r requirements.txt` 

now make sure you're in the project/av_hubert directory (cd av_hubert), then run:
```
cd fairseq

pip install --editable ./
```
### to close the virtual environment run: conda deactivate 


## Download checkpoint and files needed for preprocessing
In the project create a folder called models

place model checkpoint "fine-tuned model", shape_predictor_68_face_landmarks.dat, and 20words_mean_face.npy in this folder

You will find them in their GitHub repository `https://github.com/facebookresearch/av_hubert`
and in the demo provided.


## To run the project:
Open the project in VS code

open the terminal and run: `conda activate venv_py39`

run: `python app/api.py`

### Note: also create folders uploads and mouth_roi inside the project folder 
# ------

## To Run The code for the FIRST time 

Write the Following commands In CMD or in terminal for an empty folder in VS code 

1- git clone https://github.com/falb79/DeepfakeVideoDetection.git

2- python -m venv venv

3- venv\Scripts\activate

4- pip install -r requirements.txt

5- python -m app.main

you should be able to open http://127.0.0.1:5000 in the browser 

## To Run The code for the SECOND time 

Write these commands in project terminal

1- venv\Scripts\activate

2- python -m app.main
