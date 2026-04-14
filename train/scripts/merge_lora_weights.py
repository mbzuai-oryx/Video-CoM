import argparse
import os
import subprocess

from tqdm import tqdm


def run_merge_commands(
    input_dir,
    model_base,
    script_path="src/merge_lora_weights.py",
    use_safe_serialization=True,
):
    subdirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]
    subdirs.sort()

    for subdir in tqdm(subdirs):
        model_path = os.path.join(input_dir, subdir)
        save_model_path = os.path.join(model_path, "merged_final")

        if not os.path.exists(f"{model_path}/config.json"):
            # Model is not saved properly or the training is not completed
            continue

        if os.path.exists(save_model_path):
            # Already consolidated
            continue

        os.makedirs(save_model_path, exist_ok=True)

        cmd = [
            "python",
            script_path,
            "--model-path",
            model_path,
            "--model-base",
            model_base,
            "--save-model-path",
            save_model_path,
        ]

        if use_safe_serialization:
            cmd.append("--safe-serialization")

        print(f"🚀 Merging LoRA weights for: {model_path}")
        print(f"💾 Saving merged model to: {save_model_path}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run merge_lora_weights.py for all subdirs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing model subfolders.",
    )
    parser.add_argument(
        "--model_base", type=str, required=True, help="Path to base model checkpoint."
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="src/merge_lora_weights.py",
        help="Path to merge_lora_weights.py script.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="Do not use --safe-serialization flag.",
    )

    args = parser.parse_args()

    run_merge_commands(
        input_dir=args.input_dir,
        model_base=args.model_base,
        script_path=args.script_path,
        use_safe_serialization=not args.no_safe_serialization,
    )
