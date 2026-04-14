import os
import re

from manipulation_utils import (crop_bbox_image, get_image_from_video,
                                get_segment_time_range_list)

min_pixels = 4 * 28 * 28
max_pixels = 128 * 28 * 28
video_max_pixels = 360 * 420
nframes_segment = 16

rank = int(os.environ.get("RANK", "0"))
manipulations = ["FIND_SEGMENT", "FIND_FRAME", "SPATIAL_ZOOM"]

RESOURCE_DIR = os.environ.get("OUTPUT_DIR", "")
overlayed_videos_dir = os.environ.get("DATA_FOLDER", "")  # Dir where overlayed videos are stored
overlayed_frames_dir = f"{RESOURCE_DIR}/overlayed_frames/rank_{rank}"
overlayed_manipulations_dir = f"{RESOURCE_DIR}/overlayed_manipulations/rank_{rank}"
image_manipulations_dir = f"{RESOURCE_DIR}/image_manipulations/rank_{rank}"


os.makedirs(overlayed_frames_dir, exist_ok=True)
os.makedirs(overlayed_manipulations_dir, exist_ok=True)
os.makedirs(image_manipulations_dir, exist_ok=True)


def get_next_user_input(
    step_text, base_media, frame_number, segment_frame_list, dummy=False
):
    # if no manipulation in the step text, return the followup text
    followup_text = "Choose the final answer from the provided options."

    overlayed_video_path = base_media[0]["video"]

    if dummy:
        user_input = dummy_media_for_num_round_completion(
            overlayed_video_path, followup_text
        )
        image_content = get_image_content(user_input["image"], min_pixels, max_pixels)
        text_content = {"type": "text", "text": user_input["text"]}
        next_msg = [image_content, text_content]
        return next_msg, frame_number

    for manip in manipulations:
        if manip in step_text:
            frame_number, user_input = take_action_manipulation(
                manip,
                step_text,
                overlayed_video_path,
                frame_number,
                segment_frame_list,
                overlayed_frames_dir,
                overlayed_manipulations_dir,
            )

            if user_input is not None:
                text_content = {"type": "text", "text": user_input["text"]}
                if "video" in user_input:
                    time_range_dict = user_input["video"]["time_ranges"]
                    video_content = get_video_content(
                        overlayed_video_path,
                        nframes_segment,
                        video_max_pixels,
                        1.0,
                        time_range_dict,
                    )
                    next_msg = [video_content, text_content]

                elif "image" in user_input:
                    if isinstance(user_input["image"], list):
                        next_msg = []
                        for image in user_input["image"]:
                            next_msg.append(
                                get_image_content(image, min_pixels, max_pixels)
                            )
                        next_msg.append(text_content)
                    else:
                        image_content = get_image_content(
                            user_input["image"], min_pixels, max_pixels
                        )
                        next_msg = [image_content, text_content]
                else:
                    next_msg = [text_content]

                return next_msg, frame_number

    next_msg = [{"type": "text", "text": followup_text}]
    return next_msg, frame_number


def get_next_user_input_image_cot(step_text, base_media, dummy=False):
    # if no manipulation in the step text, return the followup text
    followup_text = "Answer the question based on the image."

    image_path = base_media[0]["image"]
    if dummy:
        user_input = {"text": followup_text, "image": image_path}
        image_content = get_image_content(user_input["image"], min_pixels, max_pixels)
        text_content = {"type": "text", "text": user_input["text"]}
        next_msg = [image_content, text_content]
        return next_msg

    if f"SPATIAL_ZOOM" in step_text:
        user_input = get_image_crop(step_text, image_path, image_manipulations_dir)
        if user_input is not None:
            text_content = {"type": "text", "text": user_input["text"]}
            if "image" in user_input:
                image_content = get_image_content(
                    user_input["image"], min_pixels, max_pixels
                )
                next_msg = [image_content, text_content]
            else:
                next_msg = [text_content]

            return next_msg

    next_msg = [{"type": "text", "text": followup_text}]
    return next_msg


def get_image_content(image_path, min_pixel, max_pixel):
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }
    return content


def get_video_content(video_path, nframes, video_max_pixels, fps, time_range_dict):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    if nframes is not None:
        content = {
            "type": "video",
            "video": video_path,
            "max_pixels": video_max_pixels,
            "nframes": nframes,
            "time_ranges": time_range_dict,
        }
    else:
        content = {
            "type": "video",
            "video": video_path,
            "max_pixels": video_max_pixels,
            "fps": fps,
            "time_ranges": time_range_dict,
        }

    return content


def dummy_media_for_num_round_completion(video_path, instruction):
    frame_number = 1
    video_name = video_path.replace(overlayed_videos_dir, "").strip("/")
    video_name = video_name.replace("/", "_")
    video_name = video_name.split(".")[0]
    image_path = os.path.join(
        overlayed_frames_dir, video_name + f"_frame_{frame_number}.jpg"
    )
    image_path = get_image_from_video(video_path, image_path, frame_number)

    if image_path and os.path.exists(image_path):
        next_message = {"text": instruction, "image": image_path}
    else:
        next_message = {"text": instruction}
    return next_message


def take_action_manipulation(
    manipulation,
    step_text,
    video_path,
    frame_number,
    segment_frame_list,
    overlayed_frames_dir,
    overlayed_manipulations_dir,
):
    next_message = None
    if manipulation == "FIND_SEGMENT":
        segment_list = []
        instruction = (
            "Continue solving the question after rewatching the selected segment."
        )
        try:
            match = re.search(
                r"FIND_SEGMENT\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                step_text,
                flags=re.IGNORECASE,
            )
            if match:
                numbers = [
                    int(x.strip())
                    for x in match.group(2).split(",")
                    if x.strip().isdigit()
                ]
                segment_list = numbers
            else:
                match = re.search(
                    r"FIND_SEGMENT\((.*?)\)\s*=\s*(\d+)", step_text, flags=re.IGNORECASE
                )
                if match:
                    segment_list = [int(match.group(2).strip())]
        except Exception as e:
            print(f"Error in FIND_SEGMENT: {e} for step_text: {step_text}")
            segment_list = []

        if len(segment_list) > 0:
            time_ranges = get_segment_time_range_list(video_path, segment_list)
            if time_ranges is None:
                next_message = {"text": instruction}
            else:
                time_range_dict = []
                for segment_range in time_ranges:
                    time_range_dict.append(
                        {"start": segment_range[0], "end": segment_range[1]}
                    )
                next_message = {
                    "text": instruction,
                    "video": {"video_path": video_path, "time_ranges": time_range_dict},
                }
                if segment_frame_list:
                    all_segment_frames = []
                    for segment_id in segment_list:
                        if segment_id in segment_frame_list:
                            frame_list = segment_frame_list[segment_id]
                            all_segment_frames.extend(frame_list)
                    if all_segment_frames:
                        next_message["video"]["frame_list"] = all_segment_frames
        else:
            next_message = {"text": instruction}

    elif manipulation == "FIND_FRAME":
        instruction = "Continue solving the question with the selected frame."
        try:
            match = re.search(
                r"FIND_FRAME\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                step_text,
                flags=re.IGNORECASE,
            )
            if match:
                numbers = [
                    int(x.strip())
                    for x in match.group(2).split(",")
                    if x.strip().isdigit()
                ]
                frame_number = numbers[0]
            else:
                match = re.search(
                    r"FIND_FRAME\((.*?)\)\s*=\s*(\d+)", step_text, flags=re.IGNORECASE
                )
                if match:
                    frame_number = int(match.group(2).strip())
        except Exception as e:
            print(f"Error in FIND_FRAME: {e} for step_text: {step_text}")
            frame_number = None
        if frame_number:
            video_name = video_path.replace(overlayed_videos_dir, "").strip("/")
            video_name = video_name.replace("/", "_")
            video_name = video_name.split(".")[0]
            image_path = os.path.join(
                overlayed_frames_dir, video_name + f"_frame_{frame_number}.jpg"
            )
            image_path = get_image_from_video(video_path, image_path, frame_number)

            if image_path and os.path.exists(image_path):
                next_message = {"text": instruction, "image": image_path}
            else:
                next_message = {"text": instruction}
        else:
            next_message = {"text": instruction}

    elif manipulation == "SPATIAL_ZOOM":
        instruction = "Continue solving the question by focusing on the zoomed region."
        video_name = video_path.replace(overlayed_videos_dir, "").strip("/")
        video_name = video_name.replace("/", "_")
        video_name = video_name.split(".")[0]
        info = []
        pattern = r"SPATIAL_ZOOM\((.*?)\)\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        try:
            for match in re.finditer(pattern, step_text, re.IGNORECASE):
                phrase = match.group(1).strip()
                crop = [int(match.group(i)) for i in range(2, 6)]
                info.append((phrase, crop))
        except Exception as e:
            print(f"Error in SPATIAL_ZOOM: {e} for step_text: {step_text}")
            info = []

        out_image_paths = []
        for phrase, crop in info:
            input_image_path = os.path.join(
                overlayed_frames_dir, video_name + f"_frame_{frame_number}.jpg"
            )
            bbox_str = bbox_str = (
                str(crop)
                .replace(", ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
            )
            manip_name = "crop_" + bbox_str
            out_image_path = input_image_path.replace(
                ".jpg", f"_manip_{manip_name}.jpg"
            )
            out_image_path = out_image_path.replace(
                overlayed_frames_dir, overlayed_manipulations_dir
            )
            out_image_path = crop_bbox_image(input_image_path, crop, out_image_path)
            out_image_paths.append(out_image_path)

        if out_image_paths:
            out_image_paths = [p for p in out_image_paths if p and os.path.exists(p)]

        if out_image_paths:
            next_message = {"text": instruction, "image": out_image_paths}
        else:
            next_message = {"text": instruction}

    return frame_number, next_message


def get_image_crop(step_text, image_path, image_manipulations_dir):
    instruction = "Continue solving the question by focusing on the zoomed region."
    try:
        m = re.search(
            r"SPATIAL_ZOOM\((.*?)\)\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
            step_text,
            re.IGNORECASE,
        )
        if not m:
            return {"text": instruction}

        bbox = [int(m.group(i)) for i in range(2, 6)]
        stem, ext = os.path.splitext(os.path.basename(image_path))
        bbox_str = str(bbox).replace("[", "").replace("]", "").replace(",", "_")
        out_image_path = os.path.join(
            image_manipulations_dir, f"{stem}_manip_crop_{bbox_str}{ext}"
        )
        out_image_path = crop_bbox_image(image_path, bbox, out_image_path)

        if out_image_path and os.path.exists(out_image_path):
            return {"text": instruction, "image": out_image_path}
    except Exception as e:
        print(f"Warning - error in get_image_crop: {e} for step_text: {step_text}")
    return {"text": instruction}
