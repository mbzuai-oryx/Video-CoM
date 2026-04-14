import copy
import os
from typing import Dict

import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                           DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
                           IGNORE_INDEX, SYSTEM_MESSAGE)
from src.params import DataArguments

from .data_utils import (get_image_info, get_video_info, llava_to_openai,
                         pad_sequence)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
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
        self.nframes_segment = 16
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        processor = self.processor

        media_items = []  # list to store all media (first turn + later turns)
        conversations = sources["conversations"]

        # ---- First turn media (image) ----
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                img, resize_info = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                )
                # Note that image is resized, use resize_info to update bbox
                media = conversations[0].get("media", None)
                media_items.append(
                    {
                        "type": "image",
                        "data": img,
                        "pixel_key": "pixel_values",
                        "grid_key": "image_grid_thw",
                        "video_kwargs": {},
                        "resize_info": resize_info,
                        "bbox": media.get("bbox", None),
                    }
                )

        # ---- First turn media (video) ----
        elif "video" in sources:
            video_files = sources["video"]
            video_folder = self.data_args.image_folder
            if isinstance(video_files, str):
                video_files = [video_files]

            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                if sources["conversations"][0].get("media", None) is not None:
                    frame_list = sources["conversations"][0]["media"].get(
                        "frame_list", []
                    )
                    frame_list = [int(i) for i in frame_list]
                else:
                    frame_list = []
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.nframes_video,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.data_args.fps,
                    frame_list,
                )
                # Mod: store video in media list
                media_items.append(
                    {
                        "type": "video",
                        "data": video_input,
                        "pixel_key": "pixel_values_videos",
                        "grid_key": "video_grid_thw",
                        "video_kwargs": video_kwargs
                        if isinstance(video_kwargs, dict)
                        else {},
                    }
                )

        # ---- First turn (Text) ----
        else:
            media_items = None

        # Skip the first because its already added in media_items
        for conv in conversations[1:]:
            if conv.get("from") != "human":
                continue
            media = conv.get("media", None)
            if isinstance(media, dict) and media["type"] == "image":
                image_folder = self.data_args.image_folder
                image_file = media["image"]
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                img, resize_info = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                )
                # Note that image is resized, use resize_info to update bbox
                media_items.append(
                    {
                        "type": "image",
                        "data": img,
                        "pixel_key": "pixel_values",
                        "grid_key": "image_grid_thw",
                        "video_kwargs": {},
                        "resize_info": resize_info,
                        "bbox": media.get("bbox", None),
                    }
                )

            elif isinstance(media, dict) and media["type"] == "video":
                time_ranges = media.get("time_ranges", None)
                start_time = media.get("start_time", None)
                end_time = media.get("end_time", None)
                frame_list = media.get("frame_list", [])
                frame_list = [int(i) for i in frame_list]
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.nframes_segment,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.data_args.fps,
                    frame_list,
                    time_ranges,
                    start_time,
                    end_time,
                )
                media_items.append(
                    {
                        "type": "video",
                        "data": video_input,
                        "pixel_key": "pixel_values_videos",
                        "grid_key": "video_grid_thw",
                        "video_kwargs": video_kwargs
                        if isinstance(video_kwargs, dict)
                        else {},
                    }
                )

        # ---- Conversation processing ----
        is_video_context = "video" in sources
        sources = copy.deepcopy(
            llava_to_openai(sources["conversations"], is_video=is_video_context)
        )

        all_input_ids = []
        all_labels = []
        all_image_pixel_values = []
        all_video_pixel_values = []
        all_image_grid_thw = []
        all_video_grid_thw = []
        all_second_grid = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        media_idx = 0  # Mod: keep track of which media item to use for this turn
        # print(sources[0]['content'])
        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                item = media_items[media_idx]
                videos = None
                images = [item["data"]]
                inputs = processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_image_pixel_values.append(inputs[item["pixel_key"]])
                all_image_grid_thw.append(inputs[item["grid_key"]])
                media_idx += 1  # Mod: increment media index for next turn

                # Mod: Resize BBox: image is resized, use resize_info to update bbox
                bboxes = item.get("bbox", None)
                resize_info = item.get("resize_info", None)
                if bboxes and resize_info:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        scale_x = resize_info["resized_w"] / resize_info["orig_w"]
                        scale_y = resize_info["resized_h"] / resize_info["orig_h"]
                        resized_bbox = [
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                        ]
                        gpt_response = gpt_response.replace(
                            str(bbox), str(resized_bbox)
                        )

            elif DEFAULT_VIDEO_TOKEN in user_input:
                item = media_items[media_idx]
                images = None
                videos = [item["data"]]
                video_kwargs = item.get("video_kwargs", {})
                inputs = processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
                all_second_grid.extend(inputs["second_per_grid_ts"])
                prompt_input_ids = inputs["input_ids"]
                all_video_pixel_values.append(inputs[item["pixel_key"]])
                all_video_grid_thw.append(inputs[item["grid_key"]])
                media_idx += 1  # Mod: increment media index for next turn

            else:
                prompt_input_ids = processor.tokenizer(
                    user_input,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"]

            response_input_ids = processor.tokenizer(
                gpt_response,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]

            input_ids = torch.cat(
                [prompt_input_ids, response_input_ids], dim=1
            ).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if len(all_image_pixel_values) > 0 and len(all_image_grid_thw) > 0:
            image_pixel_values = torch.cat(all_image_pixel_values, dim=0)
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict["pixel_values"] = image_pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        if len(all_video_pixel_values) > 0 and len(all_video_grid_thw) > 0:
            video_pixel_values = torch.cat(all_video_pixel_values, dim=0)
            video_grid_thw = torch.cat(all_video_grid_thw, dim=0)
            data_dict["pixel_values_videos"] = video_pixel_values
            data_dict["video_grid_thw"] = video_grid_thw

        if len(all_second_grid) > 0:
            second_gird = all_second_grid
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])

            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side="right", padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX
        )

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict


def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator
    )
