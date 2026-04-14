import os
import subprocess

import cv2
import numpy as np


def add_audio_to_video(original_video, video_with_overlay, final_output):
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i",
        video_with_overlay,
        "-i",
        original_video,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        final_output,
    ]
    subprocess.run(cmd, check=True)


def has_audio_stream(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            video_path,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    return "audio" in result.stdout


def get_adaptive_text_color(frame, x, y, w, h):
    roi = frame[
        max(0, y) : min(y + h, frame.shape[0]), max(0, x) : min(x + w, frame.shape[1])
    ]
    if roi.size == 0:
        return (255, 255, 255)
    mean_color = roi.mean(axis=(0, 1))
    brightness = np.mean(mean_color)
    red_dominant = (
        (mean_color[2] > 150)
        and (mean_color[2] - mean_color[1] > 50)
        and (mean_color[2] - mean_color[0] > 50)
    )
    if brightness < 150 or red_dominant:
        return (255, 255, 255)
    else:
        return (0, 0, 255)


def overlay_segment_frame_labels_all_frames(
    video_path,
    output_path,
    position="top_left",
    min_segment_duration=10,
    max_segment_duration=30,
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_sec = total_frames / fps

    initial_segments = 10
    segment_count = initial_segments

    while duration_sec / segment_count < min_segment_duration and segment_count > 1:
        segment_count -= 1

    while duration_sec / segment_count > max_segment_duration:
        segment_count += 1

    frames_per_segment = total_frames // segment_count

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        segment_id = (frame_idx // frames_per_segment) + 1
        if segment_id > segment_count:
            segment_id = segment_count

        overlay_text = f"Segment-{segment_id}\nFrame-{frame_idx + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2

        lines = overlay_text.split("\n")
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines
        ]
        total_height = sum([ts[1] + 5 for ts in text_sizes])
        max_width = max([ts[0] for ts in text_sizes])

        if position == "bottom_right":
            x = width - max_width - 10
            y = height - total_height + text_sizes[0][1]
        elif position == "top_left":
            x = 10
            y = 10 + text_sizes[0][1]
        elif position == "bottom_left":
            x = 10
            y = height - total_height + text_sizes[0][1]
        elif position == "top_right":
            x = width - max_width - 10
            y = 10 + text_sizes[0][1]
        else:
            raise ValueError(
                "Invalid position. Choose from: bottom_right, top_left, bottom_left, top_right"
            )

        for i, line in enumerate(lines):
            text_size = text_sizes[i]
            text_y = y + i * (text_size[1] + 5)
            color = get_adaptive_text_color(
                frame, x, text_y - text_size[1], text_size[0], text_size[1]
            )
            cv2.putText(
                frame,
                line,
                (x, text_y),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
