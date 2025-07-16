# Video-Content-Filtering-System-using-BERT_LSTM-and-3D-ResNet
## Table of Contents
- [Dataset](#Dataset)
- [Algorithms](#Algorithms)
- [Installation Guide](#InstallationGuide)
- [Contributing](#contributing)
- 

## **Dataset** <br>
**Profanity Dataset** <br>
The Profanity Dataset consists of positive and negative samples for text-based profanity detection. The positive dataset, containing profanity examples, was sourced from multiple repositories to ensure comprehensive coverage. <br>
- Profanity List by dsojevic
- CMU Bad Words List
- List of Dirty, Naughty, Obscene Words from LDNOOBW
- Kaggle Profanities in English Collection

The dataset includes both positive samples (containing profanity) and negative samples (clean text) to train a binary classification model. Files are provided in the repository for training and evaluation purposes.


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

Dataset Structure

kissing_dataset/

├── Folder1/ <br>
│   ├── Vid1.mp4 <br>
│   ├── Vid2.mp4 <br>
│   ├── Vid3.mp4 <br>
│   ├── Vid4.mp4 <br>
│   ├── Vid5.mp4 <br>
│   └── Annotation.json <br>

├── Folder2/ <br>
│   ├── Vid1.mp4 <br>
│   ├── Vid2.mp4 <br>
│   ├── Vid3.mp4 <br>
│   ├── Vid4.mp4 <br>
│   ├── Vid5.mp4 <br>
│   └── Annotation.json <br>
└── ... <br>

Due to CVAT free tier limitations (maximum 5 videos per project), the dataset is organized into folders containing exactly 5 videos each with corresponding JSON annotations. This structure maximizes the use of available free annotation tools while maintaining organized data management.

The kissing dataset was annotated [Here](cvat.ai)

## **Algorithms**

**Profanity Detection Model**  
- Architecture: Hybrid BERT-LSTM model for text classification  
- Purpose: Binary classification of text content (profane/clean)  
- Features: Combines BERT's contextual understanding with LSTM's sequential processing

**Kissing Detection Model** <br>
- Architecture: 3D ResNet (Residual Network) for video action recognition <br>
- Purpose: Temporal action detection in video sequences <br>
- Features: Processes spatial and temporal information simultaneously <br>

## **Installation Guide**(#InstallationGuide)
Prerequisites  
Python 3.10  
Download: [Python Official Download](https://www.python.org/downloads/)  
Ensure Python is added to PATH during installation  

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





