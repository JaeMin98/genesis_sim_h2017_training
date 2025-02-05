import os
import json
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import genesis as gs

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import (
    GOAL_ALLOWABLE_ERROR, ACTION_WEIGHT, RD_WEIGHT, RP_WEIGHT, RE_WEIGHT, RS_WEIGHT, END_SUCCESS_RATE, REPLAY_RATIO
)

class Genesis_Simulator(gym.Env):
    """
    리스트 기반의 단일 환경 Genesis 시뮬레이터 환경 클래스
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render=False, **kwargs):
        super().__init__()
        self.render_enabled = render

        self.state_dim = 9
        self.activate_joint = [0, 1, 2]  # 활성화할 관절 인덱스
        self.action_dim = len(self.activate_joint)

        # normalized space로 변경
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
        self.observation_bounds = self._load_or_explore_workspace_bounds()
        self._initialize_rl_parameters()

        self.steps = 0
        self.max_steps = 256
        
        self.Curriculum_manager = CurriculumManager()
        self.target, self.Selected_UoC = self.Curriculum_manager.get_current_target()
        self.Is_success = False

    def _initialize_scene(self):
        """씬 구성."""
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

    def _initialize_rl_parameters(self):
        self.goal_allowable_error = GOAL_ALLOWABLE_ERROR
        self.action_weight = ACTION_WEIGHT
        
        self.Rd_weight = RD_WEIGHT  # distance_reward
        self.Rp_weight = RP_WEIGHT  # progress_reward
        self.Re_weight = RE_WEIGHT  # efficiency_reward
        self.Rs_weight = RS_WEIGHT  # success_reward

    def close(self):
        gs.destroy()

    def _load_or_explore_workspace_bounds(self):
        """작업 공간 범위를 파일에서 로드하거나 새로 탐색"""
        bounds_file = "configs/observation_bounds.json"
        
        if os.path.exists(bounds_file):
            print(f"######## Loading workspace bounds from: {bounds_file}")
            with open(bounds_file, 'r') as f:
                bounds = json.loads(f.read())
            print("Workspace bounds loaded:")
            print(f"X range: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}]")
            print(f"Y range: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}]")
            print(f"Z range: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}]")
            return bounds
        else:
            bounds = self._explore_workspace_bounds()
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(bounds_file), exist_ok=True)
            
            # 결과 저장
            print(f"######## Saving workspace bounds to: {bounds_file}")
            with open(bounds_file, 'w') as f:
                json.dump(bounds, f, indent=4)
            
            return bounds

    def _explore_workspace_bounds(self, num_random_samples=10000):
        """로봇 작업 공간의 범위를 랜덤 샘플링과 체계적 탐색으로 탐색"""
        print("Exploring workspace bounds...")
        x_positions, y_positions, z_positions = [], [], []

        # 1. 체계적 탐색: 90도 간격으로 관절을 회전
        print("Systematic exploration with 90 degree intervals...")
        angles_90deg = [0, np.pi/2, np.pi, -np.pi/2]  # 0, 90, 180, -90 degrees in radians
        
        # 활성화된 각 관절에 대해 90도 간격으로 탐색
        for j1 in angles_90deg:
            for j2 in angles_90deg:
                for j3 in angles_90deg:
                    # 모든 관절을 0으로 초기화
                    joint_positions = [0.0] * len(self.joint_ranges)
                    
                    # 활성화된 관절만 설정
                    joint_positions[self.activate_joint[0]] = j1
                    joint_positions[self.activate_joint[1]] = j2
                    joint_positions[self.activate_joint[2]] = j3
                    
                    # 관절 범위 내로 클리핑
                    for idx, pos in enumerate(joint_positions):
                        low, high = self.joint_ranges[idx]
                        joint_positions[idx] = np.clip(pos, low, high)
                    
                    self.H2017.set_dofs_position(position=joint_positions)
                    self.scene.step()
                    
                    ee_pos = self.H2017.get_link("link6").get_pos()
                    if hasattr(ee_pos, 'tolist'):
                        ee_pos = ee_pos.tolist()
                    
                    x_positions.append(ee_pos[0])
                    y_positions.append(ee_pos[1])
                    z_positions.append(ee_pos[2])

        # 2. 랜덤 샘플링 탐색
        print(f"Random sampling exploration ({num_random_samples} samples)...")
        for _ in tqdm(range(num_random_samples)):
            joint_positions = self.generate_random_joint_value()
            self.H2017.set_dofs_position(position=joint_positions)
            self.scene.step()
            
            ee_pos = self.H2017.get_link("link6").get_pos()
            if hasattr(ee_pos, 'tolist'):
                ee_pos = ee_pos.tolist()
            
            x_positions.append(ee_pos[0])
            y_positions.append(ee_pos[1])
            z_positions.append(ee_pos[2])

        # 3. 경계값 계산 및 마진 추가
        margin = 0.01  # 1cm 마진 추가
        bounds = {
            'x': [min(x_positions) - margin, max(x_positions) + margin],
            'y': [min(y_positions) - margin, max(y_positions) + margin],
            'z': [min(z_positions) - margin, max(z_positions) + margin]
        }
        
        print("Workspace bounds with systematic and random sampling:")
        print(f"X range: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}]")
        print(f"Y range: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}]")
        print(f"Z range: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}]")
        return bounds

    def _normalize_joint_position(self, joint_idx, position):
        """관절 위치 정규화 (-1 ~ 1)"""
        low, high = self.joint_ranges[joint_idx]
        return 2.0 * (position - low) / (high - low) - 1.0

    def _denormalize_joint_position(self, joint_idx, normalized_position):
        """정규화된 관절 위치를 원래 범위로 변환"""
        low, high = self.joint_ranges[joint_idx]
        return low + (normalized_position + 1.0) * (high - low) / 2.0

    def _normalize_position(self, position, axis):
        """위치 좌표 정규화 (-1 ~ 1)"""
        low, high = self.observation_bounds[axis]
        return 2.0 * (position - low) / (high - low) - 1.0

    def _denormalize_position(self, normalized_position, axis):
        """정규화된 위치 좌표를 원래 범위로 변환"""
        low, high = self.observation_bounds[axis]
        return low + (normalized_position + 1.0) * (high - low) / 2.0

    def reset(self, *, seed=None, options=None):
        """
        환경 초기화. gymnasium 요구사항에 맞게 수정된 reset 메서드.
        UoC 성공률과 current_UoC를 info에 포함하여 반환
        """
        super().reset(seed=seed)  # 부모 클래스의 reset 호출
        
        # 이전 에피소드의 결과를 curriculum manager에 업데이트
        if (self.Selected_UoC != None):
            self.Curriculum_manager.update_curriculum_state(self.Is_success, self.Selected_UoC)
        self.Is_success = False

        # 관절 위치 초기화
        zero_positions = [0.0] * len(self.joint_ranges)
        self.H2017.set_dofs_position(position=zero_positions)
        
        # 새로운 목표 설정
        self._update_target()
        self.steps = 0

        obs = self._compute_state()
        
        info = {}
        self.scene.step()

        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        """
        환경 진행
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()
        
        self.last_action = action  # 현재 action 저장
        self._apply_action(action)
        self.steps += 1

        next_obs = self._compute_state()
        reward, terminated, is_success = self._compute_reward()
        truncated = self.steps >= self.max_steps

        info = {
            "is_success": is_success,
        }

        # 에피소드가 끝날 때만 Curriculum Learning 정보 추가
        if terminated or truncated:
            info.update({
                "current_uoc": self.Curriculum_manager.current_uoc,
                "all_uocs_success_rate": self.Curriculum_manager.get_all_uocs_success_rate()
            })

        return np.array(next_obs, dtype=np.float32), reward, terminated, truncated, info

    def _compute_reward(self):
        """향상된 보상 계산"""
        distance_to_goal = self.distance
        
        # 1. 거리 기반 보상 (이전보다 부드러운 형태)
        distance_reward = self.Rd_weight * (distance_to_goal ** 0.5)  # 제곱근 사용으로 거리에 따른 보상 기울기를 완화

        # 2. 진전도 보상 (이전 스텝과 비교한 진전)
        progress_reward = 0
        if hasattr(self, 'previous_distance'):
            progress = self.previous_distance - distance_to_goal
            progress_reward = self.Rp_weight * progress  # 양수면 목표에 가까워진 것, 음수면 멀어진 것
        self.previous_distance = distance_to_goal
        
        # 3. 행동 효율성 보상
        action_magnitude = sum([abs(a) for a in self.last_action]) if hasattr(self, 'last_action') else 0
        efficiency_reward = self.Re_weight * action_magnitude  # 불필요한 큰 동작 억제
        
        # 4. 성공 보상 (단계적으로)
        success_reward = 0
        is_success = distance_to_goal < self.goal_allowable_error
        if is_success:
            # 거리에 따른 단계적 보상
            if distance_to_goal < self.goal_allowable_error * 0.5:
                success_reward = self.Rs_weight*2  # 매우 정확한 도달
            else:
                success_reward = self.Rs_weight  # 기본 성공
                
        # 최종 보상 조합
        reward = (
            distance_reward +  # 기본 거리 보상
            progress_reward +  # 진전도 보상
            efficiency_reward +  # 효율성 보상
            success_reward  # 성공 보상
        )

        terminated = is_success or self.steps >= self.max_steps
        self.Is_success = is_success
        return reward, terminated, is_success

    def _compute_state(self):
        """정규화된 상태 계산"""
        # 현재 관절 위치 가져오기
        all_joints = self.get_current_joint_positions()
        if hasattr(all_joints, 'tolist'):
            all_joints = all_joints.tolist()
        
        # 활성화된 관절의 정규화된 위치 계산
        normalized_joints = [
            self._normalize_joint_position(joint_idx, all_joints[joint_idx])
            for joint_idx in self.activate_joint
        ]
        
        # 엔드이펙터 위치 정규화
        ee_pos = self.H2017.get_link("link6").get_pos()
        if hasattr(ee_pos, 'tolist'):
            ee_pos = ee_pos.tolist()
        
        normalized_ee_pos = [
            self._normalize_position(ee_pos[0], 'x'),
            self._normalize_position(ee_pos[1], 'y'),
            self._normalize_position(ee_pos[2], 'z')
        ]
        
        # 목표 위치 정규화
        normalized_target = [
            self._normalize_position(self.target[0], 'x'),
            self._normalize_position(self.target[1], 'y'),
            self._normalize_position(self.target[2], 'z')
        ]
        
        # 유클리드 거리 계산 (원래 스케일에서)
        self.distance = sum((a - b) ** 2 for a, b in zip(ee_pos, self.target[:3])) ** 0.5
        
        # 정규화된 상태 구성
        state = []
        state.extend(normalized_joints)
        state.extend(normalized_ee_pos)
        state.extend(normalized_target)
        
        return state[:self.state_dim]

    def _update_target(self):
        """새로운 목표 위치 설정"""
        self.Curriculum_manager.select_target_with_replay()
        self.target, self.Selected_UoC = self.Curriculum_manager.get_current_target()

    def _apply_action(self, action):
        """
        활성화된 관절에만 action을 적용
        """
        # 현재 관절 위치 가져오기
        current_positions = self.get_current_joint_positions()
        if hasattr(current_positions, 'tolist'):
            current_positions = current_positions.tolist()
        
        # 모든 관절에 대한 새로운 목표 위치 초기화
        target_positions = current_positions.copy()
        
        # 활성화된 관절에 대해서만 action 적용
        for i, joint_idx in enumerate(self.activate_joint):
            if i < len(action):  # action의 길이 확인
                target_positions[joint_idx] = current_positions[joint_idx] + action[i] * self.action_weight
                
                # 관절 범위 제한 적용
                low, high = self.joint_ranges[joint_idx]
                target_positions[joint_idx] = np.clip(target_positions[joint_idx], low, high)
        
        self.H2017.set_dofs_position(target_positions, self.dofs_idx)
        self.scene.step()

    def get_current_joint_positions(self):
        """현재 관절 위치 반환"""
        positions = self.H2017.get_dofs_position(dofs_idx_local=self.dofs_idx)
        return positions

    def generate_random_joint_value(self):
        """활성화된 관절에 대해서만 임의의 값 생성"""
        target_positions = [0.0] * len(self.joint_ranges)  # 모든 관절을 0으로 초기화
        
        # 활성화된 관절에 대해서만 랜덤 값 생성
        for joint_idx in self.activate_joint:
            low, high = self.joint_ranges[joint_idx]
            target_positions[joint_idx] = low + (high - low) * np.random.random()
            
        return target_positions
    
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Deque
import csv
import glob
import os
import random
from collections import defaultdict, deque

@dataclass
class CurriculumConfig:
    """Configuration settings for curriculum management"""
    REPLAY_RATIO: float = REPLAY_RATIO
    SUCCESS_THRESHOLD: float = END_SUCCESS_RATE * 0.01
    HISTORY_WINDOW_SIZE: int = 100  # Number of recent entries to consider for success rate
    EXIT_ON_SUCCESS: bool = True

class CurriculumManager:

    def __init__(self, csv_path: str = "configs/learning_contents/8000points_curriculum/data") -> None:
        self.csv_path = Path(csv_path)
        self.config = CurriculumConfig()
        self._initialize_curriculum_state()
        self.uoc_data = self._load_all_uoc_data()
        
        # Initialize success history for each UoC
        self.success_history: Dict[int, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.config.HISTORY_WINDOW_SIZE)
        )

    def _initialize_curriculum_state(self) -> None:
        """Initialize curriculum state variables"""
        self.target: List[float] = [0, 0, 0]
        self.success_rate_cache: Optional[Dict[int, float]] = None
        
        # Set UoC parameters based on available data
        self.max_uoc = self._count_csv_files()
        self.min_uoc = 1
        self.current_uoc = 1
        self.selected_uoc = 1

    def _count_csv_files(self) -> int:
        """Count number of UoC CSV files in the data directory"""
        try:
            path_pattern = os.path.join(self.csv_path, '*.csv')
            return len(glob.glob(path_pattern))
        except Exception as e:
            raise RuntimeError(f"Error accessing curriculum directory: {e}")

    def _load_all_uoc_data(self) -> List[List[List[str]]]:
        """Load all UoC data from CSV files"""
        uoc_data = []
        for uoc in range(self.min_uoc, self.max_uoc + 1):
            try:
                uoc_data.append(self._read_uoc_file(uoc))
            except Exception as e:
                raise RuntimeError(f"Error loading UoC {uoc} data: {e}")
        return uoc_data

    def _read_uoc_file(self, uoc: int) -> List[List[str]]:
        """Read individual UoC CSV file"""
        file_path = self.csv_path / f'UoC_{uoc}.csv'
        try:
            with open(file_path, 'r') as file:
                return [row[:3] for row in csv.reader(file)]
        except FileNotFoundError:
            raise FileNotFoundError(f"UoC file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading UoC file {file_path}: {e}")

    def record_episode_result(self, success: bool, uoc: int) -> None:
        """Record the result of a learning episode"""
        self.success_history[uoc].append(success)
        self.success_rate_cache = None  # Invalidate cache

    def clear_episode_history(self) -> None:
        """Clear historical episode data"""
        self.success_history.clear()
        self.success_rate_cache = None

    def calculate_success_rates(self) -> Dict[int, float]:
        """
        Calculate success rates for each UoC based on recent history.
        Uses only the most recent HISTORY_WINDOW_SIZE entries for each UoC.
        """
        if self.success_rate_cache is not None:
            return self.success_rate_cache
            
        self.success_rate_cache = {
            uoc: sum(history) / len(history)
            for uoc, history in self.success_history.items()
            if history  # Only calculate for UoCs with data
        }
        return self.success_rate_cache

    def get_success_rate(self, uoc: int) -> float:
        """Get the success rate for a specific UoC"""
        history = self.success_history[uoc]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def get_all_uocs_success_rate(self) -> list:
        all_uocs_success_rate = []
        for i in range(self.min_uoc, self.max_uoc+1):
            history = self.success_history[i]
            if not history:
                all_uocs_success_rate.append(0.0)
            else:
                success_rate = round(sum(history), 0)
                all_uocs_success_rate.append(success_rate)
        return all_uocs_success_rate
    
    def update_curriculum_state(self, success: bool, uoc: int) -> None:
        """
        Update curriculum state based on learning outcomes
        
        Args:
            success: Whether the latest episode was successful
            uoc: The UoC level of the episode
        """
        self.record_episode_result(success, uoc)
        
        # Get success rate for current UoC
        current_success_rate = self.get_success_rate(self.current_uoc)
        
        # Check if we have enough data and success rate exceeds threshold
        if (len(self.success_history[self.current_uoc]) >= self.config.HISTORY_WINDOW_SIZE and 
            current_success_rate >= self.config.SUCCESS_THRESHOLD):
            self._handle_success_progression()

    def _handle_success_progression(self) -> None:
        """Handle progression after success threshold is met"""
        next_uoc = self.current_uoc + 1
        if next_uoc <= self.max_uoc:
            self.current_uoc = next_uoc
            # self.clear_episode_history()
        elif self.config.EXIT_ON_SUCCESS:
            raise SystemExit(0)

    def select_target(self, uoc: int) -> None:
        """Select a random target from the specified UoC data"""
        if not 0 <= uoc < len(self.uoc_data):
            raise ValueError(f"Invalid UoC index: {uoc}")
            
        random_index = random.randrange(len(self.uoc_data[uoc]))
        self.target = [
            round(float(element), 4)
            for element in self.uoc_data[uoc][random_index]
        ]

    def select_target_with_replay(self) -> None:
        """Select target with replay mechanism for previous UoCs"""
        if random.random() > self.config.REPLAY_RATIO:
            self.selected_uoc = self.current_uoc
        else:
            self.selected_uoc = random.randint(1, max(1, self.current_uoc - 1))
        
        self.select_target(self.selected_uoc - 1)

    def get_current_target(self) -> Tuple[List[float], int]:
        """Get the current target and selected UoC"""
        return self.target, self.selected_uoc