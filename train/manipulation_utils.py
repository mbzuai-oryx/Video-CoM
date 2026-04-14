import os

import cv2


def get_segment_time_range_list(
    video_path, segment_list, min_segment_duration=10, max_segment_duration=30
):
    # NOTE!!!!!: Make sure the min_segment_duration and max_segment_duration are same as in overlay_segment_frame_labels_all_frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    initial_segments = 10
    segment_count = initial_segments

    while duration_sec / segment_count < min_segment_duration and segment_count > 1:
        segment_count -= 1

    while duration_sec / segment_count > max_segment_duration:
        segment_count += 1

    frames_per_segment = total_frames // segment_count
    time_ranges = []

    if len(segment_list) == segment_count:
        # Not meaningful to rewatch all segments
        print("Warning: segment_list includes all segments. Returning None.")
        return None

    for segment_number in segment_list:
        start_frame = (segment_number - 1) * frames_per_segment
        end_frame = start_frame + frames_per_segment

        if segment_number == segment_count:
            end_frame = total_frames  # Ensure last segment ends at the last frame

        start_time = start_frame / fps
        end_time = end_frame / fps

        time_ranges.append((round(start_time, 2), round(end_time, 2)))

    return time_ranges


def get_image_from_video(video_path, image_path, frame_number) -> str:
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    if os.path.exists(image_path):
        return image_path

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        print(f"Frame number {frame_number} out of range (0 to {total_frames - 1})")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        print(f"Failed to read frame {frame_number} from {video_path}")
        return None
    try:
        cv2.imwrite(image_path, frame)
    except Exception as e:
        print(f"Error saving frame {frame_number} to {image_path}: {e}")
        return None

    return image_path


def crop_bbox_image(image_path, bbox_xyxy, save_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}. Returning None.")
        return None

    if os.path.exists(save_path):
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Warning: Image not readable at {image_path}. Returning None.")
        return None

    x1, y1, x2, y2 = bbox_xyxy
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    cropped = frame[y1:y2, x1:x2]
    try:
        cv2.imwrite(save_path, cropped)
    except Exception as e:
        print(f"Error saving cropped image to {save_path}: {e}")

    return save_path
