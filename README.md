# Highway-Vehicle-Speed-Detection-
This project uses YOLOv8 for object detection and ByteTrack for multi-object tracking to estimate vehicle speeds from video footage.


## Features
- **YOLOv8 for Object Detection**: Detects vehicles in real-time.
- **ByteTrack for Object Tracking**: Tracks vehicles across frames.
- **Perspective Transformation**: Converts the scene to a top-down view for accurate speed estimation.
- **Speed Calculation**: Estimates vehicle speeds based on frame rate and distance traveled.
- **Screenshot**: Takes a screenshot of vehicles that exceed the speed of 120kmph and saves it in the data folder, each vehicle is marked with a red annotated box and a label of its speed(eg. 122 kmph)


## Usage
Run the script:
python main.py --source_video_path path/to/input/video.mp4 --target_video_path path/to/output/video.avi


### Arguments
- `--source_video_path`: Path to the input video.
- `--target_video_path`: Path to save the annotated output video.
- `--confidence_threshold`: Confidence threshold for YOLO detections (default: 0.3).
- `--iou_threshold`: IOU threshold for non-max suppression (default: 0.7).

## Output
- **Annotated Video**: The output video with bounding boxes, speed labels, and tracking IDs.
- **Screenshots**: ss of vehicles who were found speeding along with their labels

## Dependencies
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Supervision (for annotation and tracking)



https://github.com/user-attachments/assets/319c811a-dbda-40a4-a4c5-1042884b24ed


