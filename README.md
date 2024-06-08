# Crime Detection using YOLO and OpenCV


**Table of Contents**
- Introduction
- Features
- Installation
- Usage

# Introduction
This repository contains a project aimed at detecting crimes in video footage using the YOLO (You Only Look Once) object detection algorithm in combination with OpenCV, a powerful computer vision library. The main goal is to identify and flag potential criminal activities for further review.

# Features
- Real-time Crime Detection: Process video streams or recorded footage to detect potential crimes in real-time.
- Efficient and Fast: Utilizes YOLOv4, a state-of-the-art, real-time object detection system known for its speed and accuracy.
- Customizable Detection: Easily configurable to detect specific types of crimes or suspicious activities.
- OpenCV Integration: Uses OpenCV for video processing and display, making it compatible with various video formats.

## Installation
 *Prerequisites*

- Python 3.6 or higher
- OpenCV
- NumPy
- PyTorch or TensorFlow (depending on YOLO implementation)
### Steps
1. Clone the repository:

```bash
  git clone https://github.com/your-username/Crime-detection-using-YOLO-and-OpenCV.git
cd Crime-detection-using-YOLO-and-OpenCV

```
2. Create a virtual environment:

```bash
  python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
  pip install -r requirements.txt

```
4. Download the YOLO model weights:

  - Download YOLOv4 weights from YOLO's official website.
  - Place the downloaded weights in the model/ directory.

## Usage
**Running the Crime Detection Script**
1. Prepare your video file:
  Place the video file you want to process in the videos/ directory.

2. Run the script:
```bash
  python detect_crime.py --video videos/sample.mp4 --output output/processed_video.mp4

```

Replace 'sample.mp4' with your video file name. The processed video will be saved in the 'output/ directory.'

## Options
- ' --video': Path to the input video file.
- ' --output': Path to save the processed video file.
- ' --threshold': Detection confidence threshold (default is 0.5).
- ' --show': Set this flag to display the video while processing.

### Example:
```bash
python detect_crime.py --video videos/sample.mp4 --output output/result.mp4 --threshold 0.6 --show


```
