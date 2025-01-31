import os
import json
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import genesis as gs

class Genesis_Simulator(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_enabled = render

        self.state_dim = 9
        self.activate_joint = [0, 1, 2]
        self.action_dim = len(self.activate_joint)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        gs.init(backend=gs.gpu, logging_level='warning')
        self.scene = gs.Scene(
            show_viewer=self.render_enabled,
            viewer_options=gs.options.ViewerOptions(),
            rigid_options=gs.options.RigidOptions(dt=0.0001),
        )

        self._initialize_scene()
        self.learning_points_file_path = "Curriculum_builder/data_points/data_points_origin.csv"

    def _initialize_scene(self):
        self.H2017 = self.scene.add_entity(gs.morphs.MJCF(file="xml/h2017/h2017.xml"))
        self.joint_ranges = [
            [-3.1416, 3.1416],
            [-2.1819, 2.1819],
            [-2.7925, 2.7925],
            [-3.1416, 3.1416],
            [-3.1416, 3.1416],
            [-3.1416, 3.1416],
        ]
        jnt_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.dofs_idx = [self.H2017.get_joint(name).dof_idx_local for name in jnt_names]

        self.scene.build()

    def generate_target_csv(self, num_targets=20000):
        # 파일이 이미 존재하는지 확인
        if os.path.exists(self.learning_points_file_path):
            print(f"######## Target file already exists at: {self.learning_points_file_path}")
            print("######## Skipping target generation.")
            return None

        # 저장 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(self.learning_points_file_path), exist_ok=True)

        # 목표점 생성
        targets = []
        print(f"######## Generating {num_targets} target positions...")
        for _ in tqdm(range(num_targets)):
            # 현재 관절 위치 저장 (초기 위치)
            initial_joints = [0.0] * len(self.joint_ranges)
            self.H2017.set_dofs_position(position=initial_joints)
            self.scene.step()
            initial_ee_pos = self.H2017.get_link("link6").get_pos()
            if hasattr(initial_ee_pos, 'tolist'):
                initial_ee_pos = initial_ee_pos.tolist()
            
            # 랜덤 관절 위치 생성 및 적용
            joint_positions = [0.0] * len(self.joint_ranges)
            for joint_idx in self.activate_joint:
                low, high = self.joint_ranges[joint_idx]
                joint_positions[joint_idx] = low + (high - low) * np.random.random()
            
            # 새로운 관절 위치 설정
            self.H2017.set_dofs_position(position=joint_positions)
            self.scene.step()
            
            # 새로운 엔드이펙터 위치 얻기
            ee_pos = self.H2017.get_link("link6").get_pos()
            if hasattr(ee_pos, 'tolist'):
                ee_pos = ee_pos.tolist()
            
            # 거리 계산
            distance = sum((a - b) ** 2 for a, b in zip(ee_pos, initial_ee_pos)) ** 0.5
            
            # 6축과 3축 간의 델타 계산
            delta_of_6_axis = sum(abs(joint_positions[i]) for i in range(6))
            delta_of_3_axis = sum(abs(joint_positions[i]) for i in self.activate_joint)
            
            # 각 관절의 델타 계산 (초기 위치 0에서의 변화량)
            delta_joints = [abs(pos) for pos in joint_positions]
            
            # 데이터 저장
            save_data = ee_pos + [distance] + [delta_of_6_axis] + [delta_of_3_axis] + delta_joints
            targets.append(save_data)

        # CSV 파일로 저장
        # 헤더 생성
        headers = ['ee_x', 'ee_y', 'ee_z', 'distance', 'delta_6axis', 'delta_3axis']
        headers.extend([f'joint{i}_delta' for i in range(len(self.joint_ranges))])
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(targets, columns=headers)
        df.to_csv(self.learning_points_file_path, index=False)
        print(f"######## Target positions and additional data saved to: {self.learning_points_file_path}")

        return targets

    def close(self):
        gs.destroy()

def main():
    # 환경 생성
    sim = Genesis_Simulator(render=False)
    
    try:
        # 타겟 생성 (기본값: 20000개)
        # 다른 개수를 원하면 아래 숫자를 변경하세요
        sim.generate_target_csv(num_targets=20000)
    finally:
        # 환경 정리
        sim.close()

if __name__ == "__main__":
    main()