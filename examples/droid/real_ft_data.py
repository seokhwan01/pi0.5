import argparse
import numpy as np  
import json  
import os  
from pdb import set_trace 
from tqdm import tqdm
import glob  
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation as R
import random
import imageio
import shutil
from pathlib import Path

from mujoco_control.seer_dataset import validate_episode


def exists_or_mkdir(
    path
):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass 

# [x,y,z,roll,pitch,yaw] → 4x4 pose 행렬
def _6d_to_pose(
    pose6d,
    degrees=False
):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3] # 위치
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix() # 자세
    return pose

# 4x4 pose 행렬 → [x,y,z,roll,pitch,yaw]
def pose_to_6d(
    pose, 
    degrees=False
):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


# Absolute EE action을 Delta EE action으로 변환
def compute_delta_action(
    data_list,
):
    delta_cur_2_last_action_list = []
    for step_id, step_data in enumerate(data_list):
        delta_cur_2_last_action = np.zeros(7)
        # gripper action(+1/-1)은 그대로 사용
        delta_cur_2_last_action[-1] = step_data["action_gripper_pose"][-1]
        # 기준 pose 설정
        if step_id == 0: # the first timestep
            last2world = _6d_to_pose(step_data["gripper_pose"])
        else:
            last2world = _6d_to_pose(data_list[step_id-1]["action_gripper_pose"][:6], degrees=False)  

        # 현재 target pose
        cur2world = _6d_to_pose(step_data["action_gripper_pose"][:6], degrees=False)

        # 이전 pose 기준 현재 pose의 상대변환
        cur2last = np.linalg.inv(last2world) @ cur2world

        # 상대변환 → 6D delta action
        delta_cur_2_last_action[:6] = pose_to_6d(cur2last)
        delta_cur_2_last_action_list.append(delta_cur_2_last_action)

def compute_observed_delta_action(data_list):
    """관측 pose[t]에서 pose[t + 1]로 가는 Seer action을 계산한다.

    기존 ``compute_delta_action``은 controller/action_gripper_pose 기반이므로
    관측 pose 기반 전처리에는 별도 함수로 둔다. 마지막 observation은 다음
    pose가 없으므로 arm delta는 0인 terminal action으로 유지한다.
    """
    actions = []
    if not data_list:
        return actions

    observed_poses = []
    for step_data in data_list:
        pose = np.asarray(step_data["gripper_pose"], dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("invalid observed gripper_pose")
        observed_poses.append(pose)

    for step_id, step_data in enumerate(data_list):
        action = np.zeros(7, dtype=np.float32)

        if step_id + 1 < len(observed_poses):
            current2world = _6d_to_pose(observed_poses[step_id], degrees=False)
            next2world = _6d_to_pose(observed_poses[step_id + 1], degrees=False)
            next2current = np.linalg.inv(current2world) @ next2world
            action[:6] = pose_to_6d(next2current, degrees=False).astype(np.float32)

        action[-1] = float(
            np.asarray(step_data["target_gripper"]).reshape(-1)[0]
        )
        actions.append(action)

    return actions


def load_npz(path):
    """other.npz를 dictionary로 읽는다."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def save_npz(path, data):
    """dictionary를 other.npz로 저장한다."""
    np.savez_compressed(path, **data)


def get_step_dirs(episode_dir):
    """Episode의 timestep 폴더를 순서대로 반환한다."""
    return sorted(
        p for p in (Path(episode_dir) / "steps").iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def filter_episode(source_episode, destination_episode, threshold=5e-4):
    """Seer 기준으로 정지 및 미세 jitter timestep을 제거한다."""
    source_steps = get_step_dirs(source_episode)
    keep_indices = []
    previous_gripper = None

    for i, step_dir in enumerate(source_steps):
        data = load_npz(step_dir / "other.npz")
        action = np.asarray(data["delta_cur_2_last_action"], dtype=np.float64)
        gripper = float(action[-1])

        moving = (
            abs(action[0]) >= threshold
            or abs(action[1]) >= threshold
            or abs(action[2]) >= threshold
        )
        gripper_changed = (
            previous_gripper is not None
            and gripper != previous_gripper
        )

        if moving or gripper_changed:
            keep_indices.append(i)

        previous_gripper = gripper

    if not keep_indices:
        raise RuntimeError(f"No moving steps: {source_episode}")

    # 마지막 움직임 이후 timestep은 유지한다.
    last_keep = keep_indices[-1]
    keep_indices.extend(range(last_keep + 1, len(source_steps)))
    keep_indices = sorted(set(keep_indices))

    destination_steps = Path(destination_episode) / "steps"
    destination_steps.mkdir(parents=True, exist_ok=True)

    for new_id, old_id in enumerate(keep_indices):
        shutil.copytree(
            source_steps[old_id],
            destination_steps / f"{new_id:04d}",
        )

    return len(keep_indices)


def preprocess_dataset(raw_root, processed_root, dataset_name, apply_filter=False):
    """RAW dataset 전체를 Seer 학습용 processed dataset으로 변환한다."""
    raw_dataset = Path(raw_root) / dataset_name
    processed_dataset = Path(processed_root) / dataset_name

    if not raw_dataset.is_dir():
        raise FileNotFoundError(raw_dataset)

    if processed_dataset.exists():
        shutil.rmtree(processed_dataset)

    processed_dataset.mkdir(parents=True, exist_ok=True)

    exp_dirs = sorted(
        p for p in raw_dataset.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    for exp_dir in exp_dirs:
        output_exp = processed_dataset / exp_dir.name
        output_exp.mkdir(parents=True, exist_ok=True)

        episode_dirs = sorted(
            p for p in exp_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

        for episode_dir in episode_dirs:
            temp_episode = output_exp / f".{episode_dir.name}.tmp"
            final_episode = output_exp / episode_dir.name

            shutil.copytree(episode_dir, temp_episode)
            compute_episode_actions_in_episode(temp_episode)

            if apply_filter:
                filtered_episode = output_exp / f".{episode_dir.name}.filtered"

                if filtered_episode.exists():
                    shutil.rmtree(filtered_episode)

                filter_episode(temp_episode, filtered_episode)
                shutil.rmtree(temp_episode)
                os.replace(filtered_episode, final_episode)

                # Filtering changes temporal neighbors; recompute on the
                # filtered sequence.
                compute_episode_actions_in_episode(final_episode)
            else:
                os.replace(temp_episode, final_episode)

            errors = validate_episode(final_episode, processed=True)
            if errors:
                raise RuntimeError(f"{final_episode}: {'; '.join(errors)}")

            print(f"[OK] {final_episode}")

    return processed_dataset


def compute_episode_actions_in_episode(episode_dir):
    """한 episode의 NPZ에 관측 pose 기반 action을 기록한다."""
    step_dirs = get_step_dirs(episode_dir)
    if not step_dirs:
        raise RuntimeError(f"No steps: {episode_dir}")

    data_list = [load_npz(step_dir / "other.npz") for step_dir in step_dirs]
    actions = compute_observed_delta_action(data_list)

    for step_dir, data, action in zip(step_dirs, data_list, actions):
        data["delta_cur_2_last_action"] = action
        save_npz(step_dir / "other.npz", data)


def print_action_stats(processed_dataset):
    """생성된 delta action의 위치 범위를 간단히 확인한다."""
    actions = []

    for npz_path in Path(processed_dataset).glob("*/*/steps/*/other.npz"):
        data = load_npz(npz_path)
        actions.append(np.asarray(data["delta_cur_2_last_action"], dtype=np.float64))

    if not actions:
        return

    actions = np.asarray(actions)
    xyz = np.abs(actions[:, :3])
    translation = np.linalg.norm(actions[:, :3], axis=1)

    print("\nAction statistics")
    print("steps:", len(actions))
    print("mean abs xyz:", xyz.mean(axis=0))
    print("max abs xyz:", xyz.max(axis=0))
    print("max translation:", translation.max())
    print("over 0.02 m:", int(np.sum(translation > 0.02)))

# 정지 / 미세 떨림 구간 제거
def filter_real_data(
    exp_id, 
    root_path, 
    save_data_path, 
    save_gif_path
):
    root_path = os.path.join(root_path, exp_id)
    save_data_path = os.path.join(save_data_path, exp_id)
    save_gif_path = os.path.join(save_gif_path, exp_id)
    length = len(glob.glob(os.path.join(root_path, exp_id, "*")))
    exists_or_mkdir(save_gif_path)
    exists_or_mkdir(save_data_path)

    # 각 episode 처리
    for j in range(0, length): # Here we only have 100 demos, change it accordingly.
        episode_idx = str(j).zfill(6)

        # timestep별 other.npz 검색
        npz_path_list = glob.glob(os.path.join(root_path, episode_idx, "steps", "*", "other.npz"))
        npz_path_list.sort()
        step_id_list = []
        img_list = []
        for idx, npz_path in enumerate(npz_path_list):
            this_npz = np.load(npz_path)
            if idx == 0:
                prev_gripper_action = this_npz["action_gripper_pose"][-1]
            curr_gripper_action = this_npz["action_gripper_pose"][-1]
            step_id = npz_path.split('/')[-2]
            action = this_npz["delta_cur_2_last_action"]

            # XYZ 이동 또는 gripper 변화가 있으면 저장
            if (abs(action[0]) >= 5e-4) or (abs(action[1]) >= 5e-4) or (abs(action[2]) >= 5e-4) or (curr_gripper_action != prev_gripper_action):
                step_id_list.append(step_id)
            prev_gripper_action = curr_gripper_action

        # 마지막 동작 이후 데이터도 유지
        save_last_step_id = step_id_list[-1]
        last_step_id = step_id
        add_step_id_list = [str(k).zfill(4) for k in range(int(save_last_step_id)+1, int(last_step_id)+1)]
        step_id_list += add_step_id_list

        # 선택된 timestep만 새 폴더로 복사
        for new_step_id, old_step_id in tqdm(enumerate(step_id_list)):
            new_step_id = str(new_step_id).zfill(4)
            new_step_path = os.path.join(save_data_path, episode_idx, "steps", new_step_id)
            old_step_path = os.path.join(root_path, episode_idx, "steps", old_step_id)
            shutil.copytree(old_step_path, new_step_path)
            img_list.append(PILImage.open(os.path.join(new_step_path, f"image_primary.jpg")))

        # 필터링 결과 확인용 영상
        imageio.mimsave(os.path.join(save_gif_path, f"{episode_idx}.mp4"), img_list, fps=15)


# Gripper 동작 주변 데이터를 더 자주 학습하도록 index 생성
def make_aug_short_real_dataset_info(
    root_path, 
    root_info_path,
    dataset_name,
    select_ratio=1.0,
    sequence_length=7, 
    action_pred_steps=3, 
    replicate_steps=10
):
    save_json_path = os.path.join(root_info_path, f"{dataset_name}.json")
    data_list = []

    # 관측 sequence + 미래 action
    window_size = sequence_length + action_pred_steps
    exp_path_list = glob.glob(os.path.join(root_path, "*"))
    exp_path_list.sort()

    # experiment별 처리
    for exp_path in tqdm(exp_path_list):
        length = len(glob.glob(os.path.join(exp_path, "*")))

        # demo별 처리
        for j in tqdm(range(length)):
            exp_id = exp_path.split('/')[-1]
            demo_id = str(j).zfill(6)
            npz_path_list = glob.glob(os.path.join(exp_path, demo_id, "steps", "*", "other.npz"))
            npz_path_list.sort()
            this_demo_list = [f"{exp_id}/{demo_id}"]
            for npz_path in npz_path_list:
                this_npz = np.load(npz_path)
                step_id = npz_path.split('/')[-2]
                int_step_id = int(step_id)

                # 일반 학습 window 추가
                if int_step_id >= window_size:
                    this_demo_list.append([int_step_id - window_size, int_step_id])
                curr_gripper_action = this_npz["delta_cur_2_last_action"][-1]
                if step_id == "0000":
                    prev_gripper_action = curr_gripper_action

                # Gripper open/close 변화 감지
                if curr_gripper_action != prev_gripper_action:
                    print(
                        "curr_gripper_action :", curr_gripper_action, 
                        "prev_gripper_action :", prev_gripper_action,
                        "step_id :", step_id
                        )
                    # 해당 구간을 여러 번 추가
                    for _ in range(replicate_steps):
                        for k in range(action_pred_steps):
                            if int_step_id + k < len(npz_path_list):
                                this_demo_list.append([int_step_id - window_size + k, int_step_id + k])
                prev_gripper_action = curr_gripper_action
            demo_length = len(this_demo_list)
            this_demo_list.insert(1, demo_length-1+window_size)
            data_list.append(this_demo_list)

    # 전체 데이터 중 일부만 사용할 경우
    if select_ratio < 1.0:
        interval_len = 10
        start_id = 0
        select_num = int(interval_len * select_ratio)
        end_id = interval_len
        new_data_list = []
        while end_id <= len(data_list):
            selected_data_list = random.sample(data_list[start_id:end_id], select_num)
            new_data_list += selected_data_list
            start_id += interval_len
            end_id += interval_len
        data_list = new_data_list
    json_string = json.dumps(data_list, indent=1)
    # 학습용 dataset index JSON 저장
    with open(save_json_path, 'w') as json_file:
        json_file.write(json_string)


def oxe_dataset_info():
    dataset_names = [
        # {
        # "dataset_name": f"bridge_dataset",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"cmu_stretch",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        {
        "dataset_name": f"fractal20220817_data",
        "wrist_image": "Normal",
        "s_ratio": 0.54087122203,
        }, # zheng ###

        # {
        # "dataset_name": f"dlr_edan_shared_control_converted_externally_to_rlds",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"kuka",
        # "wrist_image": "Normal",
        # "s_ratio": 0.8341046294,
        # }, # zheng ###

        # {
        # "dataset_name": f"roboturk",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name": f"ucsd_kitchen_dataset_converted_externally_to_rlds",
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng

        # {
        # "dataset_name" : f"berkeley_autolab_ur5", 
        # "wrist_image": "Flip vertically & horizontally", 
        # "s_ratio": 1.0,
        # }, # fan, 
        
        # {
        # "dataset_name" : f"berkeley_fanuc_manipulation", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan

        # {
        # "dataset_name" : f"jaco_play", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"iamlab_cmu_pickup_insert_converted_externally_to_rlds", 
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng
        
        # {
        # "dataset_name" : f"viola", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"stanford_hydra_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_buds_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"utaustin_mutex", 
        # "wrist_image": "Normal",
        # "s_ratio": 1.0,
        # }, # zheng
        
        # {
        # "dataset_name" : f"taco_play", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 2.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_sailor_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"austin_sirius_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Flip vertically & horizontally",
        # "s_ratio": 1.0,
        # }, # fan
        
        # {
        # "dataset_name" : f"furniture_bench_dataset_converted_externally_to_rlds", 
        # "wrist_image": "Normal",
        # "s_ratio": 0.1,
        # }, # zheng        
    ]

    # total_data_list = []

    for info in tqdm(dataset_names):
        dataset_name = info["dataset_name"]
        wrist_image_info = info["wrist_image"]
        s_ratio = info["s_ratio"]
        root_path = f"/xxx/preprocess/oxe/{dataset_name}"
        save_json_path = f"/xxx/data_info/{dataset_name}.json"
        root_path_list = glob.glob(os.path.join(root_path, "*", "*"))
        root_path_list.sort()
        data_list = []
        data_list.append(info)
        accumulated_num_steps = 0
        for this_path in tqdm(root_path_list):
            exp_id = this_path.split('/')[-2]
            demo_id = this_path.split('/')[-1]
            num_step = len(glob.glob(os.path.join(this_path, "steps", "*")))
            if s_ratio >= 1.0:
                for _ in range(int(s_ratio)):
                    accumulated_num_steps += num_step
                    data_list.append([exp_id+'/'+demo_id, num_step])
            else:
                this_p = np.random.random()
                if this_p < s_ratio:
                    accumulated_num_steps += num_step
                    data_list.append([exp_id+'/'+demo_id, num_step])
        
        data_list[0]["accumulated_num_steps"] = accumulated_num_steps
        json_string = json.dumps(data_list, indent=1)
        with open(save_json_path, 'w') as json_file:
            json_file.write(json_string)


def main():
    parser = argparse.ArgumentParser(
        description="Create Seer's gripper-augmented training index JSON."
    )
    parser.add_argument(
        "root_path",
        help="Dataset directory containing experiment folders such as 0000.",
    )
    parser.add_argument("--root-info-path", default="./data_info")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--select-ratio", type=float, default=1.0)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--replicate-steps", type=int, default=10)
    args = parser.parse_args()

    root_path = os.path.abspath(os.path.expanduser(args.root_path))
    if not os.path.isdir(root_path):
        parser.error(f"dataset directory does not exist: {root_path}")
    dataset_name = args.dataset_name or os.path.basename(root_path.rstrip(os.sep))
    root_info_path = os.path.abspath(os.path.expanduser(args.root_info_path))
    os.makedirs(root_info_path, exist_ok=True)

    make_aug_short_real_dataset_info(
        root_path=root_path,
        root_info_path=root_info_path,
        dataset_name=dataset_name,
        select_ratio=args.select_ratio,
        sequence_length=args.sequence_length,
        action_pred_steps=args.action_pred_steps,
        replicate_steps=args.replicate_steps,
    )
    print(os.path.join(root_info_path, f"{dataset_name}.json"))


if __name__ == "__main__":
    main()
