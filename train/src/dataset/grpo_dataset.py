import copy
import os
import re
from typing import Dict

import cv2
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.constants import SYSTEM_MESSAGE
from src.params import DataArguments
from vision_process import process_vision_info


def replace_image_tokens(input_string, is_video=False):
    pattern = (
        r"\n?" + re.escape("<image>") + r"\n?"
        if not is_video
        else r"\n?" + re.escape("<video>") + r"\n?"
    )

    return re.sub(pattern, "", input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "assistant": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(
            conversation["value"], is_video=is_video
        )
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def get_image_content(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content


def get_video_content(
    video_path,
    nframes,
    video_min_pixels,
    video_max_pixels,
    width,
    height,
    fps,
    frame_list=None,
):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    if nframes is not None:
        if frame_list is not None:
            content = {
                "type": "video",
                "video": video_path,
                "min_pixels": video_min_pixels,
                "max_pixels": video_max_pixels,
                "nframes": nframes,
                "frame_list": frame_list,
            }
        else:
            content = {
                "type": "video",
                "video": video_path,
                "min_pixels": video_min_pixels,
                "max_pixels": video_max_pixels,
                "nframes": nframes,
            }
    else:
        if frame_list is not None:
            content = {
                "type": "video",
                "video": video_path,
                "max_pixels": video_max_pixels,
                "fps": fps,
                "frame_list": frame_list,
            }
        else:
            content = {
                "type": "video",
                "video": video_path,
                "max_pixels": video_max_pixels,
                "fps": fps,
            }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content


def get_first_frame_image_path(video_path):
    video_dir = video_path.split("overlayed_videos")[0]
    output_dir = video_dir + "overlayed_manipulations"
    rank = int(os.environ.get("RANK", "0"))
    output_dir = output_dir + f"/rank_{rank}"
    output_name = video_path.split("overlayed_videos")[1].split(".")[0]
    output_name = output_name.replace("/", "_") + "_first_frame.jpg"
    image_path = os.path.join(output_dir, output_name)
    if os.path.exists(image_path):
        return image_path
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    try:
        cv2.imwrite(image_path, frame)
    except Exception:
        return None
    return image_path


def resize_bbox_in_video(
    video, valid_bounding_box, min_pixel=4 * 28 * 28, max_pixel=128 * 28 * 28
):
    image_file = get_first_frame_image_path(video)
    if image_file is None:
        return []
    content = get_image_content(image_file, min_pixel, max_pixel, None, None)
    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    _, resize_info = image_input[0]
    scale_x = resize_info["resized_w"] / resize_info["orig_w"]
    scale_y = resize_info["resized_h"] / resize_info["orig_h"]

    valid_bounding_box_scaled = []
    for frame, bbox in valid_bounding_box:
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rx1 = int(round(x1 * scale_x))
            ry1 = int(round(y1 * scale_y))
            rx2 = int(round(x2 * scale_x))
            ry2 = int(round(y2 * scale_y))
            scaled_bbox = [rx1, ry1, rx2, ry2]
            valid_bounding_box_scaled.append((frame, scaled_bbox))

    return valid_bounding_box_scaled


def resize_bbox_in_image(content, bbox):
    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    _, resize_info = image_input[0]
    scale_x = resize_info["resized_w"] / resize_info["orig_w"]
    scale_y = resize_info["resized_h"] / resize_info["orig_h"]
    x1, y1, x2, y2 = bbox
    rx1 = int(round(x1 * scale_x))
    ry1 = int(round(y1 * scale_y))
    rx2 = int(round(x2 * scale_x))
    ry2 = int(round(y2 * scale_y))
    new_bbox = [rx1, ry1, rx2, ry2]
    return new_bbox


class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(GRPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.nframes_video = 32
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if "media" in sources["conversations"][0]:
            frame_list = sources["conversations"][0]["media"].get("frame_list", None)
        else:
            frame_list = None

        segment_frame_list = sources["conversations"][0].get("segment_frame_list", None)
        reasoning = sources["conversations"][-1].get("reasoning", None)
        contents = []

        if "image" in sources:
            is_video = False
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(
                    get_image_content(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                    )
                )

        elif "video" in sources:
            is_video = True
            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                if frame_list is not None:
                    contents.append(
                        get_video_content(
                            video_file,
                            self.nframes_video,
                            self.video_min_pixel,
                            self.video_max_pixel,
                            self.video_resized_w,
                            self.video_resized_h,
                            self.data_args.fps,
                            frame_list,
                        )
                    )
                else:
                    contents.append(
                        get_video_content(
                            video_file,
                            self.nframes_video,
                            self.video_min_pixel,
                            self.video_max_pixel,
                            self.video_resized_w,
                            self.video_resized_h,
                            self.data_args.fps,
                        )
                    )

        conversations = copy.deepcopy(
            llava_to_openai(sources["conversations"], is_video=is_video)
        )
        user_input = conversations[0]
        gpt_response = conversations[1]
        text_content = {"type": "text", "text": user_input["content"]}

        contents.append(text_content)

        user_prompt = [{"role": "user", "content": contents}]

        if len(SYSTEM_MESSAGE) > 0:
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)

        reasoning_plan = sources.get("plan", "")
        valid_segments = sources.get("valid_segments", [])
        valid_frames = sources.get("valid_frames", [])
        valid_bounding_box = sources.get("valid_bounding_box", [])
        valid_bounding_box_scaled = []
        if valid_bounding_box and "video" in sources:
            video_path = os.path.join(self.data_args.image_folder, sources["video"])
            valid_bounding_box_scaled = resize_bbox_in_video(
                video_path, valid_bounding_box
            )

        tloc = sources.get("tloc", [])
        bbox = sources.get("bbox", [])

        reasoning = {
            "plan": reasoning_plan,
            "valid_segments": valid_segments,
            "valid_frames": valid_frames,
            "valid_bounding_box": valid_bounding_box_scaled,
            "tloc": tloc,
            "bbox": bbox,
        }
        data_dict = dict(
            prompt=user_prompt,
            assistant=gpt_response,
            reasoning=reasoning,
            segment_frame_list=segment_frame_list,
        )

        return data_dict


def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )

    return dict(train_dataset=grpo_dataset, eval_dataset=None)
