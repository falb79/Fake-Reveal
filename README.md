## Prerequisites:
1. Download cmake - required for dlib 
2. Download ffmpeg 
3. Add both cmake and ffmpeg to system path
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


## To run the project:
Open the project in VS code

open the terminal and run: `conda activate venv_py39`

run: `app/api.py`


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
