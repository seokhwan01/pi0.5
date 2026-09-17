import numpy as np


# CSV에 저장된 gripper 값을 float로 변환
# 예: "[1.]" -> 1.0, "[-1.]" -> -1.0
def _gripper(value):
    return float(str(value).strip("[]"))


def filter_idle_rows(df):
    """로봇이 움직이지 않는 timestep을 제거한다."""
    
    # 현재 End-Effector 위치 [x, y, z]
    xyz = df[["ee_x", "ee_y", "ee_z"]].to_numpy(dtype=np.float64)
    
    # Gripper command
    gripper = np.array([_gripper(v) for v in df["target_gripper"]])

    # 이전 timestep과 비교하여 EE 위치가 변했는지 확인
    moved = np.any(xyz[1:] != xyz[:-1], axis=1)
    
    # Gripper open/close 상태가 변했는지 확인
    gripper_changed = gripper[1:] != gripper[:-1]
    
    # 첫 timestep은 항상 유지
    # 이후에는 로봇이 움직였거나 gripper가 변한 timestep만 유지
    keep = np.r_[True, moved | gripper_changed]
    
    # 선택된 timestep만 남기고 index를 0부터 다시 설정
    return df.loc[keep].reset_index(drop=True)