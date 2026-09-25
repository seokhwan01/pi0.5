from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tyro


# ============================
# Parameters
# ============================

MIN_IDLE_LEN = 15          # 1 sec @ 15Hz
MIN_NON_IDLE_LEN = 16
FILTER_LAST_N = 16         # pi0.5 horizon
VELOCITY_EPS = 1e-3

# keep frames around gripper open/close event
GRIPPER_EVENT_WINDOW = 15  # ±1 sec @ 15Hz


# ============================
# Config
# ============================

@dataclass
class Config:

    data_dir: str = str(
        Path.home() / "seokhwan/my_droid_dataset_lerobot"
    )

    action_mode: str = "custom"
    # custom:
    #   actions[:7] = commanded joint velocity [rad/s]
    #
    # droid:
    #   actions[:7] = joint velocity


# ============================
# Gripper event mask
# ============================

def get_gripper_event_mask(gripper):

    changed = (
        np.abs(
            gripper[1:] - gripper[:-1]
        ) > 1e-6
    )

    changed = np.r_[
        False,
        changed
    ]

    event_mask = np.zeros(
        len(gripper),
        dtype=bool
    )


    event_indices = np.where(
        changed
    )[0]


    for idx in event_indices:

        start = max(
            0,
            idx - GRIPPER_EVENT_WINDOW
        )

        end = min(
            len(gripper),
            idx + GRIPPER_EVENT_WINDOW + 1
        )

        event_mask[start:end] = True


    return event_mask



# ============================
# Episode filtering
# ============================

def get_keep_indices(
    episode_df,
    action_mode
):

    actions = np.stack(
        episode_df["actions"].to_numpy()
    )


    # -------------------------
    # Action interpretation
    # -------------------------

    if action_mode == "custom":

        # custom dataset
        # cmd_joint_vel [rad/s]

        joint_vel = actions[:, :7]


    elif action_mode == "droid":

        # DROID dataset
        # joint velocity action

        joint_vel = actions[:, :7]


    else:

        raise ValueError(
            f"Unknown action mode: {action_mode}"
        )


    gripper = actions[:, 7]


    # -------------------------
    # Arm idle detection
    # -------------------------

    arm_idle = np.all(
        np.abs(joint_vel)
        < VELOCITY_EPS,
        axis=1
    )


    # -------------------------
    # Gripper event protection
    # -------------------------

    gripper_event = (
        get_gripper_event_mask(
            gripper
        )
    )


    # idle only if:
    # arm stopped
    # no gripper event nearby

    is_idle = (
        arm_idle
        &
        (~gripper_event)
    )


    # -------------------------
    # Remove long idle
    # -------------------------

    padded = np.r_[
        False,
        is_idle,
        False
    ]

    diff = np.diff(
        padded.astype(int)
    )


    idle_starts = np.where(
        diff == 1
    )[0]

    idle_ends = np.where(
        diff == -1
    )[0]


    keep = np.ones(
        len(episode_df),
        dtype=bool
    )


    for start, end in zip(
        idle_starts,
        idle_ends,
        strict=True,
    ):

        if (
            end - start
        ) >= MIN_IDLE_LEN:

            keep[start:end] = False



    # -------------------------
    # Find remaining segments
    # -------------------------

    padded = np.r_[
        False,
        keep,
        False
    ]

    diff = np.diff(
        padded.astype(int)
    )


    starts = np.where(
        diff == 1
    )[0]

    ends = np.where(
        diff == -1
    )[0]


    valid = (
        (ends - starts)
        >= MIN_NON_IDLE_LEN
    )


    indices = []


    for start, end in zip(
        starts[valid],
        ends[valid],
        strict=True,
    ):

        # remove final settling frames

        end -= FILTER_LAST_N


        if end > start:

            indices.extend(
                range(start, end)
            )


    return indices



# ============================
# Main
# ============================

def main(cfg: Config):


    data_dir = Path(
        cfg.data_dir
    )


    output = (
        data_dir
        /
        "nonidle_indices.npy"
    )


    files = sorted(
        data_dir.glob(
            "data/chunk-*/episode_*.parquet"
        )
    )


    if len(files) == 0:

        raise RuntimeError(
            f"No parquet found in {data_dir}"
        )


    all_indices = []


    print(
        f"Found {len(files)} parquet files"
    )

    print(
        f"Action mode: {cfg.action_mode}"
    )


    for path in files:


        episode_df = pd.read_parquet(
            path
        )


        local_indices = (
            get_keep_indices(
                episode_df,
                cfg.action_mode
            )
        )


        # LeRobot global index

        global_indices = (
            episode_df["index"]
            .to_numpy(
                dtype=np.int64
            )
        )


        selected = (
            global_indices[
                local_indices
            ]
        )


        all_indices.extend(
            selected.tolist()
        )


        print(
            f"{path.name}: "
            f"{len(episode_df)} -> "
            f"{len(selected)}"
        )



    all_indices = np.asarray(
        all_indices,
        dtype=np.int64
    )


    np.save(
        output,
        all_indices
    )


    print()
    print(
        "Training frames:",
        len(all_indices)
    )

    print(
        "Saved:",
        output
    )



if __name__ == "__main__":

    cfg = tyro.cli(
        Config
    )

    main(cfg)