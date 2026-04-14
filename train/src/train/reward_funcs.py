import os
import re
from datetime import datetime

from src.train.reward_utils import (any_frame_in_ranges, compute_bbox_iou,
                                    extract_answer, extract_options_str,
                                    extract_pred_bbox_compact,
                                    extract_pred_frames_compact,
                                    extract_pred_segments_compact,
                                    normalize_mcq)


def accuracy_reward(completions, assistant, questions, **kwargs):
    solutions = [a["content"] for a in assistant]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, question in zip(contents, solutions, questions):
        gt_answer = sol
        options = extract_options_str(question[1]["content"])

        pred_answer = extract_answer(content)
        pred_letter = normalize_mcq(pred_answer)
        gt_letter = normalize_mcq(gt_answer)
        final_correct = pred_letter == gt_letter

        if final_correct:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


def reasoning_reward(completions, assistant, reasoning_gts, questions, **kwargs):
    rewards = []
    solutions = [a["content"] for a in assistant]
    contents = [completion[0]["content"] for completion in completions]
    for content, sol, gt_info, question in zip(
        contents, solutions, reasoning_gts, questions
    ):
        pred_reasoning = content
        pred_answer = extract_answer(content)
        gt_answer = sol

        gt_letter = normalize_mcq(gt_answer)
        pred_letter = normalize_mcq(pred_answer)
        final_correct = pred_letter == gt_letter
        accuracy_score = 1.0 if final_correct else 0.0

        valid_segments = gt_info.get("valid_segments", [])
        valid_frames = gt_info.get("valid_frames", [])
        valid_bboxes = gt_info.get("valid_bounding_box", [])

        pred_segs = extract_pred_segments_compact(pred_reasoning)
        pred_frames = extract_pred_frames_compact(pred_reasoning)
        pred_bbox_pairs = extract_pred_bbox_compact(pred_reasoning)

        gt_bb_map = {int(f): bb for f, bb in valid_bboxes} if valid_bboxes else {}

        if valid_segments and pred_segs:
            seg_correct_count = sum(1 for s in pred_segs if s in valid_segments)
            seg_total = len(pred_segs)
        else:
            seg_correct_count = 0
            seg_total = 0

        if valid_frames and pred_frames:
            frame_correct_count = sum(
                1 for f in pred_frames if any_frame_in_ranges([f], valid_frames)
            )
            frame_total = len(pred_frames)
        else:
            frame_correct_count = 0
            frame_total = 0

        bbox_correct_count = 0
        bbox_total = 0
        if pred_bbox_pairs and gt_bb_map:
            for frame, bbox in pred_bbox_pairs:
                if frame is None:
                    continue
                if int(frame) in gt_bb_map:
                    bbox_total += 1
                    iou = compute_bbox_iou(bbox, gt_bb_map[int(frame)])
                    if iou >= 0.35:
                        bbox_correct_count += 1

        correct_sum = seg_correct_count + frame_correct_count + bbox_correct_count
        total_sum = seg_total + frame_total + bbox_total
        reasoning_score = (correct_sum / total_sum) if total_sum > 0 else 0.0

        final_score = accuracy_score + reasoning_score
        rewards.append(final_score)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Match any text (including newlines) that ends with the FINAL_ANSWER line
    pattern = r"(?s).*FINAL_ANSWER:\s*[A-Z]\.\s*.+$"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content.strip()) for content in completion_contents
    ]
    rewards = [1.0 if match else 0.0 for match in matches]

    return rewards
