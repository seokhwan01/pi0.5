"""
Minimal example script for converting a dataset collected on the DROID platform to LeRobot format.

Usage:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro
# from droid_csv_preprocess import filter_idle_rows

# REPO_NAME = "your_hf_username/my_droid_dataset"  # Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "seokhwan/my_droid_dataset"

    # Observation:
    #     - RGB image
    #     - joint position
    #     - gripper position

    # Action:
    #     - joint velocity (7D)
    #     - gripper command (1D)

def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    # output_path = HF_LEROBOT_HOME / REPO_NAME
    output_path = Path.home() / "my_droid_dataset_lerobot"
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir = Path(data_dir)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            # "exterior_image_2_left": {
            #     "dtype": "image",
            #     "shape": (180, 320, 3),
            #     "names": ["height", "width", "channel"],
            # },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_paths = sorted([p for p in data_dir.iterdir() if p.is_dir() and (p / "steps.csv").exists()])

    print(f"Found {len(episode_paths)} episodes for conversion")

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        episode_df = pd.read_csv(episode_path / "steps.csv")
        
        # episode_df = filter_idle_rows(episode_df)
        
        # 2. 여기서 joint velocity 계산
        joint_cols = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

        joint_pos = episode_df[joint_cols].to_numpy(dtype=np.float64)
        timestamps = episode_df["timestamp"].to_numpy(dtype=np.float64)

        joint_vel = np.zeros_like(joint_pos)

        dt = timestamps[1:] - timestamps[:-1]
        
        if np.any(dt <= 0):
            raise ValueError(f"Non-positive dt found in {episode_path}")


        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt[:, None]

        joint_vel[-1] = joint_vel[-2]
        
        

        for i, row in episode_df.iterrows():
            primary_path = episode_path / row["primary"]
            wrist_path = episode_path / row["wrist"]
            
            
            # --------------------------------------------------
            # DROID observation gripper_position
            # 0 = open, 1 = closed
            # --------------------------------------------------
            finger_1 = float(row["finger_1"])
            finger_2 = float(row["finger_2"])
            gripper_width = finger_1 + finger_2
            gripper_position = 1.0 - (gripper_width / 0.08)
            gripper_position = float(np.clip(gripper_position, 0.0, 1.0))
            
            primary = np.array(Image.open(primary_path).convert("RGB"))
            wrist = np.array(Image.open(wrist_path).convert("RGB"))

            primary = resize_image(primary, (320, 180))
            wrist = resize_image(wrist, (320, 180))

            raw_gripper = float(
                str(row["target_gripper"]).strip("[]")
            )

            action_gripper = (1.0 - raw_gripper) / 2.0
            action_gripper = float(np.clip(action_gripper, 0.0, 1.0))

            language_instruction = str(row["language"])

            if language_instruction.startswith("b'") and language_instruction.endswith("'"):
                language_instruction = language_instruction[2:-1]
                
            action = np.concatenate(
                [joint_vel[i], np.asarray([action_gripper], dtype=np.float64),]).astype(np.float32)

            dataset.add_frame(
                {
                    "exterior_image_1_left": primary,
                    "wrist_image_left": wrist,
                    "joint_position": np.asarray(
                        [
                            row["joint_1"],
                            row["joint_2"],
                            row["joint_3"],
                            row["joint_4"],
                            row["joint_5"],
                            row["joint_6"],
                            row["joint_7"],
                        ],
                        dtype=np.float32,
                    ),
                    "gripper_position": np.asarray([gripper_position], dtype=np.float32),
                    "actions": action,
                    "task": language_instruction,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
