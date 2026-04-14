import re
import string
from statistics import mean

from rouge_score import rouge_scorer


def extract_answer(text):
    answer = text.split("FINAL_ANSWER:")[-1].strip()
    return answer


def extract_reasoning(text):
    reasoning = text.split("FINAL_ANSWER:")[0].strip()
    return reasoning


def extract_options_str(prompt):
    question = [p for p in prompt if p["type"] == "text"][0]["text"]
    options = question.split("Options:")[-1].strip()
    options = options.split("\n")[0].strip()
    return options


def normalize_mcq(ans: str) -> str:
    """Extract just the option letter (A-E) if present."""
    match = re.match(r"([A-Ea-e])", ans.strip())
    if match:
        return match.group(1).upper()
    return ans.strip()


# Manipulation Extraction utils for Compact
def extract_pred_segments_compact(step_text):
    steps = step_text.split("\n")
    vals = []
    try:
        for step in steps:
            if "FIND_SEGMENT" in step:
                match = re.search(
                    r"FIND_SEGMENT\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                    step,
                    flags=re.IGNORECASE,
                )
                if match:
                    numbers = [
                        int(x.strip())
                        for x in match.group(2).split(",")
                        if x.strip().isdigit()
                    ]
                    vals.extend(numbers)
                    continue
                match = re.search(
                    r"FIND_SEGMENT\((.*?)\)\s*=\s*(\d+)", step, flags=re.IGNORECASE
                )
                if match:
                    numbers = [int(match.group(1).strip())]
                    vals.extend(numbers)
        unique_vals = list(set(vals))
    except Exception as e:
        print(
            f"Warning - error in extract_pred_segments_compact: {e} for step_text: {step_text}"
        )
        unique_vals = []
    return unique_vals


def extract_pred_frames_compact(step_text):
    steps = step_text.split("\n")
    vals = []
    try:
        for step in steps:
            if "FIND_FRAME" in step:
                match = re.search(
                    r"FIND_FRAME\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                    step,
                    flags=re.IGNORECASE,
                )
                if match:
                    numbers = [
                        int(x.strip())
                        for x in match.group(2).split(",")
                        if x.strip().isdigit()
                    ]
                    vals.extend(numbers)
                    continue
                match = re.search(
                    r"FIND_FRAME\((.*?)\)\s*=\s*(\d+)", step, flags=re.IGNORECASE
                )
                if match:
                    numbers = [int(match.group(2).strip())]
                    vals.extend(numbers)
        unique_vals = list(set(vals))
    except Exception as e:
        print(
            f"Warning - error in extract_pred_frames_compact: {e} for step_text: {step_text}"
        )
        unique_vals = []
    return unique_vals


def extract_pred_bbox_compact(step_text):
    # sanity check if this extraction is required
    if "SPATIAL_ZOOM" not in step_text:
        return []
    steps = step_text.split("\n")
    pairs = []
    current_frame = None
    try:
        for step in steps:
            if "FIND_FRAME" in step:
                match = re.search(
                    r"FIND_FRAME\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                    step,
                    flags=re.IGNORECASE,
                )
                if match:
                    numbers = [
                        int(x.strip())
                        for x in match.group(2).split(",")
                        if x.strip().isdigit()
                    ]
                    if numbers:
                        current_frame = numbers[-1]  # use the last frame on that line
                else:
                    match = re.search(
                        r"FIND_FRAME\((.*?)\)\s*=\s*(\d+)", step, flags=re.IGNORECASE
                    )
                    if match:
                        current_frame = int(match.group(2).strip())
            if "SPATIAL_ZOOM" in step:
                m = re.search(
                    r"SPATIAL_ZOOM\((.*?)\)\s*=\s*\[([0-9,\s]+)\]",
                    step,
                    flags=re.IGNORECASE,
                )
                if m:
                    nums = [
                        int(x.strip())
                        for x in m.group(2).split(",")
                        if x.strip().isdigit()
                    ]
                    for i in range(0, len(nums) - 3, 4):
                        bbox = [nums[i], nums[i + 1], nums[i + 2], nums[i + 3]]
                        pairs.append((current_frame, bbox))
                m = re.search(
                    r"SPATIAL_ZOOM\((.*?)\)\s*=\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)",
                    step,
                    flags=re.IGNORECASE,
                )
                if m:
                    bbox = [
                        int(m.group(2)),
                        int(m.group(3)),
                        int(m.group(4)),
                        int(m.group(5)),
                    ]
                    pairs.append((current_frame, bbox))
                m = re.search(
                    r"SPATIAL_ZOOM\((.*?)\)\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                    step,
                    flags=re.IGNORECASE,
                )
                if m:
                    bbox = [
                        int(m.group(2)),
                        int(m.group(3)),
                        int(m.group(4)),
                        int(m.group(5)),
                    ]
                    pairs.append((current_frame, bbox))
    except Exception as e:
        print(
            f"Warning - error in extract_pred_bbox_compact: {e} for step_text: {step_text}"
        )
        pairs = []
    return pairs


# Reasoning utils for VQA
def any_frame_in_ranges(frames, ranges_):
    for f in frames:
        for s, e in ranges_:
            if s <= f <= e:
                return True
    return False


def compute_bbox_score(pred_bbox_pairs, valid_bounding_box):
    if not pred_bbox_pairs or not valid_bounding_box:
        mean_iou = 0.0
        return mean_iou
    gt_map = {int(f): bb for f, bb in valid_bounding_box}
    scores = []
    for frame, pr_bb in pred_bbox_pairs:
        if frame is None:
            continue
        frame = int(frame)
        if frame not in gt_map:
            continue
        gt_bb = gt_map[frame]
        iou = compute_bbox_iou(pr_bb, gt_bb)
        if iou >= 0.5:
            loc_score = 1.0
        elif iou >= 0.3:
            loc_score = 0.5
        elif iou >= 0.1:
            loc_score = 0.25
        else:
            loc_score = 0.0
        scores.append(loc_score)
    mean_iou = mean(scores) if scores else 0.0
    return mean_iou


# Temporal Localization utils
def compute_temporal_iou(gt, pred):
    try:
        gt_start, gt_end = gt
        pred_start, pred_end = pred
        inter_start = max(gt_start, pred_start)
        inter_end = min(gt_end, pred_end)
        intersection = max(0, inter_end - inter_start)
        union = max(gt_end, pred_end) - min(gt_start, pred_start)
        iou = intersection / union if union > 0 else 0
        iou = round(iou, 2)
    except Exception as e:
        print(f"Warning - error in compute_temporal_iou: {e}")
        return 0.0
    return iou


def extract_frame_range(text):
    try:
        m = re.search(
            r"\bframe[s]?\s*:?[\s\-]*?(\d+)(?:\s*(?:to|[-–—])\s*(\d+))?",
            text,
            re.IGNORECASE,
        )
        if not m:
            return [0, 0]
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        frame_range = [start, end]
    except Exception as e:
        print(f"Warning - error in extract_frame_range: {e} for text: {text}")
        frame_range = [0, 0]
    return frame_range


# Spatial Localization utils
def compute_bbox_iou(boxA, boxB):
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    except Exception as e:
        print(f"Warning - error in compute_bbox_iou: {e}")
        iou = 0.0
    return iou


def extract_answer_oe(text):
    answer = text.split("FINAL_ANSWER:")[-1].strip()
    return answer


def extract_text_after_bbox(text: str):
    i = text.rfind("]")
    answer = text[i + 1 :].strip() if i != -1 else text.strip()
    return answer


def compute_rouge_score_oe(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3
    return average_fmeasure


# Relaxed MCQ Accuracy Utils
def _normalize_text(s):
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    s = re.sub(r"\s+", " ", s)
    return s


def _split_letter_and_text(s):
    s = s.strip()
    m = re.match(r"^([A-Ea-e])\s*\.?\s*(.*)$", s)
    if m:
        letter = m.group(1).upper()
        text = m.group(2).strip()
        return letter, text
    return "", s


def _build_option_maps(options):
    letter_to_full = {}
    letter_to_text = {}
    normtext_to_full = {}
    for opt in options:
        m = re.match(r"^\s*([A-Ea-e])\s*\.\s*(.*)$", opt.strip())
        if not m:
            continue
        letter = m.group(1).upper()
        text = m.group(2).strip()
        letter_to_full[letter] = f"{letter}. {text}"
        letter_to_text[letter] = text
        normtext_to_full[_normalize_text(text)] = f"{letter}. {text}"
    return letter_to_full, letter_to_text, normtext_to_full


def parse_options(options_str):
    pattern = re.compile(
        r"(?<!\w)([A-Ea-e])\.\s*(.*?)\s*(?=(?<!\w)[A-Ea-e]\.\s*|$)", re.S
    )
    out = []
    for m in pattern.finditer(options_str):
        letter = m.group(1).upper()
        text = m.group(2).strip().rstrip(",.")
        out.append(f"{letter}. {text}")
    return out


def check_final_correct_mcq(answer, sol, options_str):
    options = parse_options(options_str)
    letter_to_full, letter_to_text, normtext_to_full = _build_option_maps(options)
    sol_letter, sol_text = _split_letter_and_text(sol)
    ans_letter, ans_text = _split_letter_and_text(answer)
    sol_full = letter_to_full.get(sol_letter, sol.strip())
    ans_full_by_letter = letter_to_full.get(ans_letter, "")
    ans_full_by_text = normtext_to_full.get(_normalize_text(ans_text), "")
    if _normalize_text(
        ans_full_by_letter or ans_full_by_text or answer
    ) == _normalize_text(sol_full):
        return True
    if ans_letter and sol_letter and ans_letter == sol_letter:
        return True
    if (
        _normalize_text(ans_text)
        and _normalize_text(sol_text)
        and _normalize_text(ans_text) == _normalize_text(sol_text)
    ):
        return True
    return False


def normalize_number(num_str):
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception as e:
        print(f"Error converting '{num_str}' to float: {e}")
        return None


def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[m][n] / max(1, m)


def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3
    return average_fmeasure


def infer_question_type(answer):
    ans = answer.strip()

    # Rule: multiple choice (letter optionally followed by '.' or ')' and text)
    if re.match(r"^[A-Ea-e][\.\)]?\s*(.*)", ans):
        return "multiple choice"

    # Rule: numerical
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})*(\.\d+)?", ans) or re.fullmatch(
        r"[-+]?\d+(\.\d+)?", ans
    ):
        if "." in ans and len(ans.split(".")[-1]) > 3:
            return "regression"
        return "numerical"

    if len(ans.split()) <= 5 and re.fullmatch(r"[A-Za-z0-9\s\-:]+", ans):
        return "OCR"

    return "free-form"
