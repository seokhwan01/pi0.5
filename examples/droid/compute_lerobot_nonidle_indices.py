from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path.home() / "my_droid_dataset_lerobot"
OUTPUT = DATA_DIR / "nonidle_indices.npy"

MIN_IDLE_LEN = 7
MIN_NON_IDLE_LEN = 16
FILTER_LAST_N = 10
VELOCITY_EPS = 1e-3


def get_keep_indices(episode_df):
    actions = np.stack(episode_df["actions"].to_numpy())

    joint_vel = actions[:, :7]
    gripper = actions[:, 7]

    # 팔이 거의 움직이지 않는 frame
    arm_idle = np.all(
        np.abs(joint_vel) < VELOCITY_EPS,
        axis=1,
    )

    # gripper가 바뀌는 frame은 유지
    gripper_changed = np.r_[
        False,
        np.abs(gripper[1:] - gripper[:-1]) > 1e-6,
    ]

    is_idle = arm_idle & ~gripper_changed

    # 연속 idle 구간 찾기
    padded = np.r_[False, is_idle, False]
    diff = np.diff(padded.astype(int))

    idle_starts = np.where(diff == 1)[0]
    idle_ends = np.where(diff == -1)[0]

    # 7 frame 이상 idle인 구간만 제거
    long_idle = (
        idle_ends - idle_starts
    ) >= MIN_IDLE_LEN

    keep = np.ones(len(episode_df), dtype=bool)

    for start, end in zip(
        idle_starts[long_idle],
        idle_ends[long_idle],
        strict=True,
    ):
        keep[start:end] = False

    # 남은 연속 구간 찾기
    padded = np.r_[False, keep, False]
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    valid = (
        ends - starts
    ) >= MIN_NON_IDLE_LEN

    indices = []

    for start, end in zip(
        starts[valid],
        ends[valid],
        strict=True,
    ):
        end -= FILTER_LAST_N

        if end > start:
            indices.extend(range(start, end))

    return indices


def main():
    files = sorted(
        DATA_DIR.glob(
            "data/chunk-*/episode_*.parquet"
        )
    )

    all_indices = []

    for path in files:
        episode_df = pd.read_parquet(path)

        local_indices = get_keep_indices(
            episode_df
        )

        # LeRobot이 저장한 실제 global index 사용
        global_indices = (
            episode_df["index"]
            .to_numpy(dtype=np.int64)
        )

        selected = global_indices[
            local_indices
        ]

        all_indices.extend(selected.tolist())

        print(
            f"{path.name}: "
            f"{len(episode_df)} -> "
            f"{len(selected)}"
        )

    all_indices = np.asarray(
        all_indices,
        dtype=np.int64,
    )

    np.save(OUTPUT, all_indices)

    print()
    print("학습에 사용할 frame:", len(all_indices))
    print("저장:", OUTPUT)


if __name__ == "__main__":
    main()