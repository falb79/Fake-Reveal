## Prerequisites:
### 1. **Install cmake - required for dlib**
   
   Download from:  `https://cmake.org/download/`

   **Choose:** Windows x64 Installer or macOS
   
   **Note:** Enable *Add to PATH* during installation.  

### 2. **Install ffmpeg**

   Download from: `https://www.gyan.dev/ffmpeg/builds/`
   
   **Choose:** `ffmpeg-git-full.7z`
   
   **Note:** Unzip the folder, then use PowerShell (Run as Administrator) to add FFmpeg to SYSTEM PATH

   
### 3. **Install Conda:**
   
   Install Anaconda or Miniconda (if not already installed).

### 4. **Install Build Tools** 

   Download: `https://visualstudio.microsoft.com/visual-cpp-build-tools/`
   
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
### 3. Create a virtual environment using conda, and install requirements  
In a new terminal run:
```
conda create -n venv_py39 python=3.9

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

**Note**: If you face permission issues, run Anaconda CMD as Administrator.


## Download checkpoint and files needed for preprocessing

Create a folder inside the project named (models) and add:

   - Fine-tuned model

   - shape_predictor_68_face_landmarks.dat

   - 20words_mean_face.npy

These files can be found in the AV-HuBERT repository `https://github.com/facebookresearch/av_hubert` or the demo files provided.

## Required Additional Folders

📂Inside the project directory, create these empty folders:

   - uploads
   
   - mouth_roi



## To run the project:
Open the project in VS Code, then in the terminal:
```
conda activate venv_py39

python app/api.py
```

**Note:** to close the virtual environment run: conda deactivate 


