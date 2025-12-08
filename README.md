# Fake Reveal

Fake-Reveal is an AI-powered platform dedicated to detecting and revealing deepfake videos.

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

## Citations
We acknowledge and are grateful for the foundational work provided by the following open-source projects and research papers.

1- AV-HuBERT:

@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184}
    year = {2022}
}

@article{shi2022avsr,
    author  = {Bowen Shi and Wei-Ning Hsu and Abdelrahman Mohamed},
    title = {Robust Self-Supervised Audio-Visual Speech Recognition},
    journal = {arXiv preprint arXiv:2201.01763}
    year = {2022}
}

2. ViT Deepfake Image Detection :
To enhance user experience, we used the API from Dima806's project on Kaggle for image classification.
[Deepfake vs real faces detection ViT](https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit/notebook)


