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
    GOAL_ALLOWABLE_ERROR, ACTION_WEIGHT, RD_WEIGHT, RP_WEIGHT, RE_WEIGHT, RS_WEIGHT
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
        self.generate_target_csv()

        self.steps = 0
        self.max_steps = 256

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
        """목표 위치 로드 및 초기화."""
        self.learning_points_file_path = "configs/learning_contents/learning_points.csv"
        df = pd.read_csv(self.learning_points_file_path, header=None)
        self.target_file_data = df.values.tolist()
        self.target = [0.0] * len(self.target_file_data[0])

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
        """
        super().reset(seed=seed)  # 부모 클래스의 reset 호출
        
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

        info = {"is_success": is_success}

        return np.array(next_obs, dtype=np.float32), reward, terminated, truncated, info

    def _compute_reward(self):
        """보상 계산"""
        distance_to_goal = self.distance
        reward = - distance_to_goal

        is_success = distance_to_goal < self.goal_allowable_error
        if is_success:
            reward += 100

        terminated = is_success or self.steps >= self.max_steps

        return reward, terminated, is_success

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
        random_idx = np.random.randint(0, len(self.target_file_data))
        self.target = self.target_file_data[random_idx]

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
            # 모든 관절을 0으로 초기화
            joint_positions = [0.0] * len(self.joint_ranges)
            
            # 활성화된 관절에 대해서만 랜덤 값 생성
            for joint_idx in self.activate_joint:
                low, high = self.joint_ranges[joint_idx]
                joint_positions[joint_idx] = low + (high - low) * np.random.random()
            
            # 조인트 위치 설정
            self.H2017.set_dofs_position(position=joint_positions)
            self.scene.step()
            
            # 엔드이펙터 위치 얻기
            ee_pos = self.H2017.get_link("link6").get_pos()
            if hasattr(ee_pos, 'tolist'):
                ee_pos = ee_pos.tolist()
            
            targets.append(ee_pos)

        # CSV 파일로 저장
        df = pd.DataFrame(targets)
        df.to_csv(self.learning_points_file_path, index=False, header=False)
        print(f"######## Target positions saved to: {self.learning_points_file_path}")

        return targets

import time
from datetime import datetime
import numpy as np

def test_actions_with_report(env, save_report=True):
    """
    Genesis_Simulator 환경의 action 적용을 상세히 테스트하고 결과를 문서화하는 함수
    
    Args:
        env: Genesis_Simulator 인스턴스
        save_report: 결과를 파일로 저장할지 여부
    """
    report_lines = []
    def log(message):
        print(message)
        report_lines.append(message)

    log(f"\n{'='*80}")
    log(f"Genesis Simulator 상세 테스트 리포트")
    log(f"테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*80}\n")

    # 1. 환경 설정 정보
    log("1. 환경 설정 정보")
    log(f"- 활성화된 관절: {env.activate_joint}")
    log(f"- 상태 차원: {env.state_dim}")
    log(f"- 행동 차원: {env.action_dim}")
    log(f"- Action 가중치: {env.action_weight}")
    log(f"- 최대 스텝 수: {env.max_steps}")
    log(f"- 목표 거리 임계값: {env.goal_allowable_error}")
    
    for idx, (low, high) in enumerate(env.joint_ranges):
        log(f"- 관절 {idx} 범위: [{low:.4f}, {high:.4f}] rad")
    log("")

    # 2. 초기화 상태 테스트
    log("2. 초기화 상태 테스트")
    obs, _ = env.reset()
    initial_joints = env.get_current_joint_positions()
    if hasattr(initial_joints, 'tolist'):
        initial_joints = initial_joints.tolist()
    
    log("- 초기 관절 위치:")
    for idx, pos in enumerate(initial_joints):
        log(f"  관절 {idx}: {pos:.6f} rad")
    
    ee_pos = env.H2017.get_link("link6").get_pos()
    if hasattr(ee_pos, 'tolist'):
        ee_pos = ee_pos.tolist()
    log(f"- 초기 엔드이펙터 위치: [{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}]")
    log("")

    # 3. 개별 관절 테스트
    log("3. 개별 관절 테스트")
    for joint_idx in env.activate_joint:
        log(f"\n테스트 대상: 관절 {joint_idx}")
        
        # 환경 초기화
        env.reset()
        initial_pos = env.get_current_joint_positions()
        if hasattr(initial_pos, 'tolist'):
            initial_pos = initial_pos.tolist()
        
        # 단일 관절 동작 테스트
        action = [0.0] * env.action_dim
        action_idx = env.activate_joint.index(joint_idx)
        action[action_idx] = 1.0
        
        log(f"- 적용할 action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_pos = env.get_current_joint_positions()
        if hasattr(new_pos, 'tolist'):
            new_pos = new_pos.tolist()
        
        change = new_pos[joint_idx] - initial_pos[joint_idx]
        log(f"- 관절 변화량: {change:.6f} rad")
        log(f"- 예상 변화량: {env.action_weight:.6f} rad")
        log(f"- 오차: {abs(change - env.action_weight):.6f} rad")
    log("")

    # 4. 연속 동작 테스트
    log("4. 연속 동작 테스트")
    env.reset()
    test_sequence = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0]
    ]
    
    for i, action in enumerate(test_sequence):
        log(f"\n동작 {i+1}")
        log(f"- Action: {action}")
        
        initial_pos = env.get_current_joint_positions()
        if hasattr(initial_pos, 'tolist'):
            initial_pos = initial_pos.tolist()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_pos = env.get_current_joint_positions()
        if hasattr(new_pos, 'tolist'):
            new_pos = new_pos.tolist()
        
        log("- 관절 변화:")
        for idx in env.activate_joint:
            change = new_pos[idx] - initial_pos[idx]
            log(f"  관절 {idx}: {change:.6f} rad")
    log("")

    # 5. 경계값 테스트
    log("5. 경계값 테스트")
    env.reset()
    test_actions = [
        ([1.0] * env.action_dim, "최대 양의 action"),
        ([-1.0] * env.action_dim, "최대 음의 action"),
        ([2.0] * env.action_dim, "범위 초과 양의 action"),
        ([-2.0] * env.action_dim, "범위 초과 음의 action")
    ]
    
    for action, desc in test_actions:
        log(f"\n{desc}")
        log(f"- Action: {action}")
        
        initial_pos = env.get_current_joint_positions()
        if hasattr(initial_pos, 'tolist'):
            initial_pos = initial_pos.tolist()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_pos = env.get_current_joint_positions()
        if hasattr(new_pos, 'tolist'):
            new_pos = new_pos.tolist()
        
        log("- 관절 상태:")
        for idx in env.activate_joint:
            low, high = env.joint_ranges[idx]
            within_limits = low <= new_pos[idx] <= high
            log(f"  관절 {idx}: {new_pos[idx]:.6f} rad (제한범위 내: {within_limits})")
    log("")

    # 6. 비활성화된 관절 안정성 테스트
    log("6. 비활성화된 관절 안정성 테스트")
    env.reset()
    inactive_joints = [i for i in range(len(env.joint_ranges)) if i not in env.activate_joint]
    
    initial_pos = env.get_current_joint_positions()
    if hasattr(initial_pos, 'tolist'):
        initial_pos = initial_pos.tolist()
    
    # 여러 번의 랜덤 동작 수행
    n_random_actions = 1000
    max_deviation = 0.0
    
    log(f"- {n_random_actions}회의 랜덤 동작 테스트 수행")
    for i in range(n_random_actions):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, terminated, truncated, info = env.step(action.tolist())
        
        current_pos = env.get_current_joint_positions()
        if hasattr(current_pos, 'tolist'):
            current_pos = current_pos.tolist()
        
        for joint_idx in inactive_joints:
            deviation = abs(current_pos[joint_idx] - initial_pos[joint_idx])
            max_deviation = max(max_deviation, deviation)
    
    log(f"- 비활성화된 관절들의 최대 편차: {max_deviation:.8f} rad")
    log(f"- 안정성 판정: {'안정' if max_deviation < 1e-6 else '불안정'}")
    log("")

    # 7. 종합 평가
    log("7. 종합 평가")
    issues = []
    
    # Action 가중치 정확도 평가
    if abs(change - env.action_weight) > 1e-6:
        issues.append("- Action 가중치가 정확하게 적용되지 않음")
    
    # 비활성화된 관절 안정성 평가
    if max_deviation > 1e-6:
        issues.append("- 비활성화된 관절이 움직임")
    
    # 관절 제한 평가
    for idx in env.activate_joint:
        if not (low <= new_pos[idx] <= high):
            issues.append(f"- 관절 {idx}가 제한 범위를 벗어남")
    
    if issues:
        log("\n발견된 문제점:")
        for issue in issues:
            log(issue)
    else:
        log("모든 테스트 항목이 정상적으로 통과되었습니다.")
    
    # 리포트 저장
    if save_report:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'action_test_report_{timestamp}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n테스트 리포트가 {filename}에 저장되었습니다.")

    return report_lines

def test_state_and_reward(env, report_lines):
    """
    Genesis_Simulator 환경의 상태와 보상을 테스트하고 결과를 문서화하는 함수
    
    Args:
        env: Genesis_Simulator 인스턴스
        report_lines: 기존 리포트 라인 리스트
    """
    def log(message):
        print(message)
        report_lines.append(message)

    # 8. 상태(State) 테스트
    log("\n8. 상태(State) 테스트")
    
    # 8.1 상태 정규화 테스트
    log("\n8.1 상태 정규화 테스트")
    obs, _ = env.reset()
    
    log("- 상태 구성 요소:")
    log(f"  * 상태 차원: {env.state_dim}")
    log(f"  * 활성화된 관절 위치 (정규화): {obs[:len(env.activate_joint)]}")
    log(f"  * 엔드이펙터 위치 (정규화): {obs[len(env.activate_joint):len(env.activate_joint)+3]}")
    log(f"  * 목표 위치 (정규화): {obs[-3:]}")
    
    # 정규화 범위 확인
    within_range = all(-1.0 <= x <= 1.0 for x in obs)
    log(f"- 모든 상태 값이 [-1, 1] 범위 내: {within_range}")

    # 8.2 상태 일관성 테스트
    log("\n8.2 상태 일관성 테스트")
    n_test_steps = 100
    state_consistency_issues = []
    
    for _ in range(n_test_steps):
        action = np.random.uniform(-1, 1, env.action_dim)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 상태 벡터 길이 확인
        if len(next_obs) != env.state_dim:
            state_consistency_issues.append(f"상태 벡터 길이 불일치: {len(next_obs)} != {env.state_dim}")
        
        # 정규화 범위 확인
        if not all(-1.0 <= x <= 1.0 for x in next_obs):
            state_consistency_issues.append("정규화 범위 벗어남")

    if state_consistency_issues:
        log("\n발견된 상태 일관성 문제:")
        for issue in state_consistency_issues:
            log(f"- {issue}")
    else:
        log("- 상태 일관성 테스트 통과")

    # 9. 보상(Reward) 테스트
    log("\n9. 보상(Reward) 테스트")
    
    # 9.1 보상 구성요소 테스트
    log("\n9.1 보상 구성요소 테스트")
    env.reset()
    log(f"- 거리 보상 가중치 (Rd_weight): {env.Rd_weight}")
    log(f"- 진전도 보상 가중치 (Rp_weight): {env.Rp_weight}")
    log(f"- 효율성 보상 가중치 (Re_weight): {env.Re_weight}")
    log(f"- 성공 보상 가중치 (Rs_weight): {env.Rs_weight}")
    
    # 9.2 보상 범위 테스트
    log("\n9.2 보상 범위 테스트")
    rewards = []
    distances = []
    n_reward_tests = 1000
    
    for _ in range(n_reward_tests):
        action = np.random.uniform(-1, 1, env.action_dim)
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
        distances.append(env.distance)
    
    log(f"- 보상 통계:")
    log(f"  * 최소 보상: {min(rewards):.4f}")
    log(f"  * 최대 보상: {max(rewards):.4f}")
    log(f"  * 평균 보상: {np.mean(rewards):.4f}")
    log(f"  * 표준편차: {np.std(rewards):.4f}")
    
    log(f"- 거리 통계:")
    log(f"  * 최소 거리: {min(distances):.4f}")
    log(f"  * 최대 거리: {max(distances):.4f}")
    log(f"  * 평균 거리: {np.mean(distances):.4f}")

    # 9.3 보상 연속성 테스트
    log("\n9.3 보상 연속성 테스트")
    env.reset()
    prev_reward = None
    reward_jumps = []
    
    for _ in range(100):
        action = np.random.uniform(-1, 1, env.action_dim)
        _, reward, _, _, _ = env.step(action)
        
        if prev_reward is not None:
            reward_change = abs(reward - prev_reward)
            reward_jumps.append(reward_change)
        
        prev_reward = reward
    
    max_jump = max(reward_jumps)
    avg_jump = np.mean(reward_jumps)
    log(f"- 보상 변화 통계:")
    log(f"  * 최대 보상 변화: {max_jump:.4f}")
    log(f"  * 평균 보상 변화: {avg_jump:.4f}")
    
    # 9.4 목표 달성 보상 테스트
    log("\n9.4 목표 달성 보상 테스트")
    env.reset()
    success_rewards = []
    n_success_tests = 100
    
    # 다양한 관절 위치에서 목표 달성 테스트
    for _ in range(n_success_tests):
        env.reset()  # 새로운 목표 위치로 리셋
        target_pos = env.target[:3]
        
        # 랜덤한 관절 위치로 여러번 시도
        best_distance = float('inf')
        best_reward = None
        best_success = False
        
        # 각 테스트에서 여러 번의 랜덤 시도
        for attempt in range(50):  # 50번의 랜덤 시도
            # 랜덤한 관절 위치 생성
            random_joints = env.generate_random_joint_value()
            env.H2017.set_dofs_position(random_joints)
            env.scene.step()
            
            # 현재 엔드이펙터 위치 확인
            ee_pos = env.H2017.get_link("link6").get_pos()
            if hasattr(ee_pos, 'tolist'):
                ee_pos = ee_pos.tolist()
            
            # 목표까지의 거리 계산
            distance = sum((a - b) ** 2 for a, b in zip(ee_pos, target_pos)) ** 0.5
            
            if distance < best_distance:
                best_distance = distance
                _, reward, _, _, info = env.step([0, 0, 0])  # 제자리 동작으로 보상 계산
                best_reward = reward
                best_success = info.get('is_success', False)
        
        success_rewards.append((best_reward, best_success))
    
    successes = sum(1 for _, success in success_rewards if success)
    log(f"- 목표 도달 테스트:")
    log(f"  * 성공률: {successes/n_success_tests*100:.2f}%")
    log(f"  * 성공 시 평균 보상: {np.mean([r for r, s in success_rewards if s]):.4f}")
    
    return report_lines

def run_complete_test(env, save_report=True):
    """
    Genesis_Simulator의 전체 테스트를 실행하는 함수
    """
    # 기존 테스트 실행
    report_lines = test_actions_with_report(env, save_report=False)
    
    # 상태와 보상 테스트 추가
    report_lines = test_state_and_reward(env, report_lines)
    
    # 리포트 저장
    if save_report:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'complete_test_report_{timestamp}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n테스트 리포트가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    
    # 환경 생성
    simulation = Genesis_Simulator(render=False)

    # 환경 체크
    check_env(simulation, warn=True)
    simulation.step([0.1,0.1,0.1])
    print(simulation.reset())
    # run_complete_test(simulation)

    simulation.close()