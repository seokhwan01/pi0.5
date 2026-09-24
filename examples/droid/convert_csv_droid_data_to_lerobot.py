"""
Convert custom Panda CSV dataset to LeRobot format.

Usage:

Custom action:
uv run examples/droid/convert_csv_droid_data_to_lerobot.py \
    --data-dir ~/my_droid_dataset/blue_box_red_zone \
    --action-mode custom

DROID-style action:
uv run examples/droid/convert_csv_droid_data_to_lerobot.py \
    --data-dir ~/my_droid_dataset/blue_box_red_zone \
    --action-mode droid

Output:
~/my_droid_dataset_lerobot

Action definition:

custom:
    actions[0:7] = cmd_joint_vel_1 ~ cmd_joint_vel_7
                 = commanded/reference joint velocity [rad/s]

droid:
    actions[0:7] = droid_joint_vel_1 ~ droid_joint_vel_7
                 = DROID-style joint action saved by the collector

both:
    actions[7] = target_gripper
               = 0 open, 1 close
"""

from pathlib import Path
from typing import Literal
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro


REPO_NAME = "seokhwan/my_droid_dataset"


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(
        image.resize(
            size,
            resample=Image.BICUBIC,
        )
    )


def main(
    data_dir: str,
    *,
    action_mode: Literal["custom", "droid"],
    push_to_hub: bool = False,
):
    # --------------------------------------------------
    # Input / Output
    # --------------------------------------------------
    data_dir = Path(data_dir).expanduser()

    output_path = (
        Path.home()
        / "my_droid_dataset_lerobot"
    )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    # 기존 변환 결과가 있으면 삭제 후 새로 생성
    if output_path.exists():
        shutil.rmtree(output_path)

    print(
        f"Input dataset : {data_dir}"
    )
    print(
        f"Output dataset: {output_path}"
    )
    print(
        f"Action mode   : {action_mode}"
    )

    # --------------------------------------------------
    # LeRobot dataset
    # --------------------------------------------------
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": [
                    "height",
                    "width",
                    "channel",
                ],
            },

            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": [
                    "height",
                    "width",
                    "channel",
                ],
            },

            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": [
                    "joint_position"
                ],
            },

            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": [
                    "gripper_position"
                ],
            },

            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": [
                    "actions"
                ],
            },
        },

        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --------------------------------------------------
    # Episode folders
    #
    # blue_box_red_zone/
    # ├── 000000/
    # │   ├── steps.csv
    # │   ├── primary/
    # │   └── wrist/
    # ├── 000001/
    # ...
    # --------------------------------------------------
    episode_paths = sorted(
        [
            p
            for p in data_dir.iterdir()
            if (
                p.is_dir()
                and (p / "steps.csv").exists()
            )
        ]
    )

    print(
        f"Found {len(episode_paths)} "
        f"episodes for conversion"
    )

    if len(episode_paths) == 0:
        raise RuntimeError(
            f"No episodes containing steps.csv "
            f"found in {data_dir}"
        )

    # --------------------------------------------------
    # Convert episodes
    # --------------------------------------------------
    for episode_path in tqdm(
        episode_paths,
        desc="Converting episodes",
    ):
        episode_df = pd.read_csv(
            episode_path / "steps.csv"
        )

        if len(episode_df) == 0:
            raise ValueError(
                f"Empty steps.csv: {episode_path}"
            )

        # ==================================================
        # Select ARM ACTION representation
        # ==================================================
        if action_mode == "custom":
            # ----------------------------------------------
            # Custom
            #
            # 수집 당시 controller reference velocity
            # 실제 commanded joint velocity [rad/s]
            #
            # q[t+1] - q[t]로 다시 계산하지 않는다.
            # ----------------------------------------------
            arm_action_cols = [
                "cmd_joint_vel_1",
                "cmd_joint_vel_2",
                "cmd_joint_vel_3",
                "cmd_joint_vel_4",
                "cmd_joint_vel_5",
                "cmd_joint_vel_6",
                "cmd_joint_vel_7",
            ]

        elif action_mode == "droid":
            # ----------------------------------------------
            # DROID-style
            #
            # Collector에서 이미 계산해서 저장한
            # droid_joint_vel_* 값을 그대로 사용
            #
            # 다시 변환하지 않는다.
            # ----------------------------------------------
            arm_action_cols = [
                "droid_joint_vel_1",
                "droid_joint_vel_2",
                "droid_joint_vel_3",
                "droid_joint_vel_4",
                "droid_joint_vel_5",
                "droid_joint_vel_6",
                "droid_joint_vel_7",
            ]

        else:
            # Literal 때문에 실제로 여기 들어올 일은 없음
            raise ValueError(
                f"Unknown action_mode: {action_mode}"
            )

        # 필요한 action column 존재 확인
        missing_cols = [
            col
            for col in arm_action_cols
            if col not in episode_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing action columns in "
                f"{episode_path / 'steps.csv'}: "
                f"{missing_cols}"
            )

        # --------------------------------------------------
        # Arm action [N, 7]
        # --------------------------------------------------
        arm_actions = episode_df[
            arm_action_cols
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            arm_actions
        ).all():
            raise ValueError(
                f"Non-finite arm action found "
                f"in {episode_path}"
            )

        # ==================================================
        # Frames
        # ==================================================
        for i, row in episode_df.iterrows():

            primary_path = (
                episode_path
                / row["primary"]
            )

            wrist_path = (
                episode_path
                / row["wrist"]
            )

            # --------------------------------------------------
            # Observation:
            # gripper_position
            #
            # Franka:
            # finger1 + finger2 = 전체 gripper width
            #
            # width = 0.08 m -> fully open
            # width = 0.00 m -> fully closed
            #
            # DROID/OpenPI representation:
            # 0 = open
            # 1 = closed
            # --------------------------------------------------
            finger_1 = float(
                row["finger_1"]
            )

            finger_2 = float(
                row["finger_2"]
            )

            gripper_width = (
                finger_1
                + finger_2
            )

            gripper_position = (
                1.0
                - (
                    gripper_width
                    / 0.08
                )
            )

            gripper_position = float(
                np.clip(
                    gripper_position,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Images
            # --------------------------------------------------
            primary = np.array(
                Image.open(
                    primary_path
                ).convert("RGB")
            )

            wrist = np.array(
                Image.open(
                    wrist_path
                ).convert("RGB")
            )

            # 수집 이미지가 이미 320x180이므로 resize 불필요
            #
            # primary = resize_image(
            #     primary,
            #     (320, 180),
            # )
            #
            # wrist = resize_image(
            #     wrist,
            #     (320, 180),
            # )

            # --------------------------------------------------
            # Gripper ACTION
            #
            # Custom / DROID 둘 다 동일
            #
            # 0 = OPEN
            # 1 = CLOSE
            # --------------------------------------------------
            raw_gripper = float(
                row["target_gripper"]
            )

            action_gripper = float(
                np.clip(
                    raw_gripper,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Language instruction
            # --------------------------------------------------
            language_instruction = str(
                row["language"]
            )

            if (
                language_instruction.startswith("b'")
                and language_instruction.endswith("'")
            ):
                language_instruction = (
                    language_instruction[2:-1]
                )

            # ==================================================
            # Final 8D ACTION
            #
            # custom:
            #   [cmd_joint_vel(7), gripper(1)]
            #
            # droid:
            #   [droid_joint_vel(7), gripper(1)]
            # ==================================================
            action = np.concatenate(
                [
                    arm_actions[i],

                    np.asarray(
                        [action_gripper],
                        dtype=np.float64,
                    ),
                ]
            ).astype(
                np.float32
            )

            # --------------------------------------------------
            # Add frame
            # --------------------------------------------------
            dataset.add_frame(
                {
                    "exterior_image_1_left":
                        primary,

                    "wrist_image_left":
                        wrist,

                    "joint_position":
                        np.asarray(
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

                    "gripper_position":
                        np.asarray(
                            [
                                gripper_position
                            ],
                            dtype=np.float32,
                        ),

                    "actions":
                        action,

                    "task":
                        language_instruction,
                }
            )

        dataset.save_episode()

    # --------------------------------------------------
    # Optional Hugging Face upload
    # --------------------------------------------------
    if push_to_hub:
        dataset.push_to_hub(
            tags=[
                "libero",
                "panda",
                "rlds",
            ],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print()
    print("==============================")
    print("Conversion completed")
    print(f"Action mode: {action_mode}")
    print(f"Output: {output_path}")
    print("==============================")


if __name__ == "__main__":
    tyro.cli(main)