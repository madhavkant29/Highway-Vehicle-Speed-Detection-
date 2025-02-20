# Highway-Vehicle-Speed-Detection-
This project uses YOLOv8 for object detection and ByteTrack for multi-object tracking to estimate vehicle speeds from video footage.


## Features
- **YOLOv8 for Object Detection**: Detects vehicles in real-time.
- **ByteTrack for Object Tracking**: Tracks vehicles across frames.
- **Perspective Transformation**: Converts the scene to a top-down view for accurate speed estimation.
- **Speed Calculation**: Estimates vehicle speeds based on frame rate and distance traveled.


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
- **Screenshots**: Images of vehicles exceeding 120 km/h.

## Dependencies
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Supervision (for annotation and tracking)

