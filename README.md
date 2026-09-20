# Content-Based Video Retrieval System

A university project for **content-based video search using deep learning**. The system preprocesses videos into scenes, selects representative keyframes, generates natural-language captions for them, and lets a user search for relevant video segments through a small Flask web application.

The project was developed for the *Image and Video Analysis with Deep Learning* course at the University of Klagenfurt.

## What the project does

The pipeline has two main parts:

1. **Video preprocessing**
   - detects scene boundaries with **TransNetV2**
   - extracts frames with **FFmpeg**
   - selects a representative keyframe using perceptual hashing and a saliency-based score
   - generates a caption for the selected keyframe with **BLIP**
   - stores scene timestamps, captions and keyframe paths in an **SQLite** database

2. **Video search**
   - converts a text query into an embedding using **SentenceTransformers**
   - embeds the stored scene captions
   - searches for the closest caption using **FAISS**
   - returns the matching video segment and displays it in the browser

The web interface also contains optional support for submitting retrieved segments to **DRES**.

## Technologies

- Python
- PyTorch
- TransNetV2
- Hugging Face Transformers / BLIP
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS
- OpenCV
- FFmpeg
- SQLite
- Flask
- HTML / JavaScript

## Repository structure

```text
.
├── app.py                 # Flask application and semantic search
├── source_code.ipynb      # video preprocessing pipeline
├── templates/
│   └── index.html         # web interface
├── README.md
└── video_processing_results.db   # generated after preprocessing
```

The preprocessing step also creates folders containing detected scenes, keyframes and a visualization of the detected shot boundaries.

## How it works

```text
Video
  ↓
TransNetV2 scene detection
  ↓
Frame extraction
  ↓
Representative keyframe selection
  ↓
BLIP image captioning
  ↓
SQLite database
  ↓
SentenceTransformer embeddings + FAISS
  ↓
Text query → most relevant video scene
```

## Running the project

The original project was developed with a local copy of the V3C video dataset, so a few paths in the code need to be adapted before running it on another machine.

### 1. Install the dependencies

The main Python dependencies are:

```bash
pip install torch transformers sentence-transformers flask faiss-cpu     numpy pillow opencv-python imagehash ffmpeg-python requests
```

FFmpeg itself also needs to be installed and available from the command line.

### 2. Add TransNetV2

The notebook expects the TransNetV2 implementation and pretrained weights to be available locally. In particular, the code loads:

```text
transnetv2-pytorch-weights.pth
```

Update the import or model paths if your TransNetV2 installation is stored elsewhere.

### 3. Configure the video dataset

Change the local video directory used in the notebook and Flask application to the location of your own videos.

The original version used a structure similar to:

```text
V3C1-100/
├── 00104/
│   └── 00104.mp4
├── 00105/
│   └── 00105.mp4
└── ...
```

### 4. Preprocess the videos

Run `source_code.ipynb`. It will detect scenes, create keyframes and captions, and populate:

```text
video_processing_results.db
```

### 5. Start the web application

Place `index.html` inside a `templates/` directory and run:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

Enter a natural-language query such as *"a person walking outside"* and the application will return the scene whose generated caption is closest to the query.

## Notes

This repository contains the implementation created for the course project rather than a production-ready application. The video dataset itself is not included, and some paths/configuration are specific to the environment in which the project was developed.

## Authors

- Mahmoud Hamed
- Kirill Feigelman
- Andrii Zhukov
- Begench Batyrov
