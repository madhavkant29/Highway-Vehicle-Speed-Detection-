import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hardcoded parameters
source_video_path = "data/vehicles.mp4"
target_video_path = "data/vehicles-result.mp4"
confidence_threshold = 0.3
iou_threshold = 0.5

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
model = YOLO("yolov8x.pt")

byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
)

frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

polygon_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

first_frame = next(frame_generator)
height, width = first_frame.shape[:2]

video_info = sv.VideoInfo(width=width, height=height, fps=video_info.fps)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(target_video_path, fourcc, video_info.fps, (width, height))

try:
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=iou_threshold)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Convert to km/h
                labels.append(f"#{tracker_id} {int(speed)} km/h")
                if speed > 120:
                    screenshot_filename = f"screenshot_{tracker_id}.jpg"
                    idx = np.where(detections.tracker_id == tracker_id)[0]
                    if len(idx) > 0:
                        x1, y1, x2, y2 = detections.xyxy[idx[0]]
                        screenshot_frame = frame.copy()
                        cv2.rectangle(
                            screenshot_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),
                            3,
                        )
                        cv2.putText(
                            screenshot_frame,
                            f"{int(speed)} km/h",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imwrite(screenshot_filename, screenshot_frame)
                        print(
                            f"Screenshot saved: {screenshot_filename}, Speed: {int(speed)} km/h"
                        )

        annotated_frame = frame.copy()
        annotated_frame = sv.TraceAnnotator(
            thickness=2, trace_length=video_info.fps * 2
        ).annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.BoxAnnotator(thickness=2).annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = sv.LabelAnnotator(
            text_scale=1, text_thickness=2, text_position=sv.Position.BOTTOM_CENTER
        ).annotate(scene=annotated_frame, detections=detections, labels=labels)

        out.write(annotated_frame)
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    out.release()
    cv2.destroyAllWindows()