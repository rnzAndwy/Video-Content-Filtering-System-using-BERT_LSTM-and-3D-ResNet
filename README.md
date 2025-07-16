# Video-Content-Filtering-System-using-BERT_LSTM-and-3D-ResNet
## Table of Contents
- [Dataset](#Dataset)
- [Pre-trained Models](#Pre-trained-Models)
- [Algorithms](#Algorithms)
- [Installation Guide](#Installation-Guide)
- [Environment Set-up](#Environment-Set-up)
- [Application Set-up](#Application-Set-up)
- [Verification](#Verification)

## **Dataset** <br>
**Profanity Dataset** <br>
The Profanity Dataset consists of positive and negative samples for text-based profanity detection. The positive dataset, containing profanity examples, was sourced from multiple repositories to ensure comprehensive coverage. <br>
- Profanity List by dsojevic
- CMU Bad Words List
- List of Dirty, Naughty, Obscene Words from LDNOOBW
- Kaggle Profanities in English Collection

The dataset includes both positive samples (containing profanity) and negative samples (clean text) to train a binary classification model. Files are provided in the repository for training and evaluation purposes.

Access Dataset: [Profanity Dataset](https://huggingface.co/datasets/rnzandwy/Profanity_Dataset)


**Kissing Dataset** <br>
The Kissing Scene Dataset was manually curated to ensure accuracy and relevance for video action recognition. The dataset consists of annotated video frames extracted from movie clips, with each video clip containing exactly 96 frames to maintain consistency. (Though this was already configured in the code)

- Positive samples: Annotated frames where 15% of the frames depict explicit kissing scenes
- Negative samples: Unannotated frames with no explicit romantic content

Annotation Process:

The dataset was manually annotated based on a clear definition of "kissing" action. However, several limitations may affect model performance:
- Limited diversity in samples (demographics, contexts, angles)
- Varying video quality across sources
- Inconsistent lighting conditions
- Cultural variations in romantic expressions

Access Dataset: [Kissing Dataset](https://huggingface.co/datasets/rnzandwy/Kissing_Dataset)

Dataset Structure
```
kissing_dataset/
├── Folder1/ 
│   ├── Vid1.mp4 
│   ├── Vid2.mp4 
│   ├── Vid3.mp4 
│   ├── Vid4.mp4
│   ├── Vid5.mp4 
│   └── Annotation.json 

├── Folder2/ <br>
│   ├── Vid1.mp4 
│   ├── Vid2.mp4 
│   ├── Vid3.mp4 
│   ├── Vid4.mp4 
│   ├── Vid5.mp4 
│   └── Annotation.json 
└── ... 
```

Due to CVAT free tier limitations (maximum 5 videos per project), the dataset is organized into folders containing exactly 5 videos each with corresponding JSON annotations. This structure maximizes the use of available free annotation tools while maintaining organized data management.

The kissing dataset was annotated [Here](https://www.cvat.ai/)

# **Pre-trained Models**  
Pre-trained models are available for both profanity detection and kissing scene detection tasks.  
Access Models: Pre-trained Models Repository  
Model Structure
```
models/
├── BERT_LSTM/
│   ├── model files
│   └── configuration files
└── ResNet/
    ├── model files
    └── configuration files
```
Required Folder Names:  
BERT_LSTM - Contains the trained BERT-LSTM model for profanity detection  
ResNet - Contains the trained 3D ResNet model for kissing scene detection  

**Training Pipeline and Annotations**
If you want to examine the annotations and explore the training pipeline:  
1. Navigate to the training_pipeline folder in the repository  
2. Open the relevant Jupyter notebooks for the model you're interested in  
3. Update the file paths in the notebooks to point to where your dataset is stored on your computer  
4. Run the notebook cells to execute the training pipeline  

This allows you to:
- Review the annotation process and data structure
- Understand the model training methodology
- Retrain models with your own datasets
- Experiment with different hyperparameters

## **Algorithms**

**Profanity Detection Model**  
- Architecture: Hybrid BERT-LSTM model for text classification  
- Purpose: Binary classification of text content (profane/clean)  
- Features: Combines BERT's contextual understanding with LSTM's sequential processing

**Kissing Detection Model** <br>
- Architecture: 3D ResNet (Residual Network) for video action recognition <br>
- Purpose: Temporal action detection in video sequences <br>
- Features: Processes spatial and temporal information simultaneously <br>

## **Installation Guide**
**Prerequisites**  
Python 3.10  
Download: [Python Official Download](https://www.python.org/downloads/)  
Ensure Python is added to PATH during installation  

Node.js and npm
Download: [Node.js Official Download](https://nodejs.org/)  
Download the LTS version (includes npm)  
Verify installation by running: node --version and npm --version  

Anaconda  
Download: [Anaconda Distribution](https://www.anaconda.com/products/distribution)  
Provides package management and virtual environment capabilities  

CUDA Toolkit & NVIDIA Drivers  
Download: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
Download: [NVIDIA Drivers](https://www.nvidia.com/drivers/)  
Required for GPU acceleration  

FFmpeg  
Download: [FFmpeg Official](https://ffmpeg.org/download.html)  
Essential for video processing operations  

## **Environment Set-up**  
**Setting System Environment Variables**
Navigate to System Properties → Advanced → Environment Variables  
Add the following paths to your PATH variable:    
```
C:\Users\[username]\anaconda3\Scripts
C:\Users\[username]\anaconda3
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\ffmpeg\bin
C:\Program Files\nodejs\
```
Note: *Adjust paths according to your installation directories*  

**CUDA Compatibility Check**  
Before proceeding, verify compatibility between your CUDA version and deep learning frameworks:  
[PyTorch](https://pytorch.org/get-started/locally/)  
[TensorFlow](https://www.tensorflow.org/install/source#gpu)  


**Setting up Anaconda Environment**  
Option 1: Manual Environment Creation  
Create virtual environment:  
```  
conda create -n your_env python=3.10
conda activate your_env
```

Register kernel for Jupyter notebooks:  
```
conda install ipykernel
python -m ipykernel install --user --name your_env --display-name "your_env"
```

Install required libraries:
```
# Deep Learning Frameworks
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorflow[and-cuda]

# NLP Libraries
pip install transformers tokenizers datasets

# Computer Vision
pip install opencv-python pillow

# Data Processing
pip install pandas numpy matplotlib seaborn

# Jupyter and Development
conda install jupyter notebook
pip install tqdm scikit-learn

# Video Processing
pip install moviepy
```

Option 2: Automated Environment Setup (Recommended)  
Two automated options are provided for convenience:  

Using YAML file:  
```
conda env create -f final_environment.yml
conda activate tensorr1 # tensorr1 was the name of the environment 
```

Using batch file:  
```
# If conda is not initialized in your terminal
conda init

# Run the automated setup
dev-start
```
The dev-start.bat file will automatically create the environment with all necessary libraries installed and configured.  

# **Application Set-up**  
Installing Node.js Dependencies  
After setting up the Python environment, you need to install the Node.js dependencies for the Electron application:  

Navigate to the project root directory:  
```
cd path/to/Video-Content-Filtering-System-using-BERT_LSTM-and-3D-ResNet
```
Install Node.js dependencies:  
```
npm install
```
Important Notes:  
Do NOT run npm run build or npm build as this will package the entire system unnecessarily  
Only run npm install to install the required node_modules  
The node_modules folder and package.json are required for the Electron application to function properly  

## Verification  
After installation, verify your set-up:  
```
# Activate environment
conda activate your_env

# Check Python version
python --version

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")
```






