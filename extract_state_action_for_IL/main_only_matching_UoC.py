import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from GenesisEnv_CL import Genesis_Simulator
from tqdm import tqdm
import glob

ENV = Genesis_Simulator(render=False)
DETERMINISTIC = False

class ModelValidator:
    def __init__(self, model_path, num_episodes_per_uoc=100, save_failed_episodes=False, max_steps_per_episode=None, output_dir="demonstration_inform"):
        self.model_path = model_path
        self.num_episodes_per_uoc = num_episodes_per_uoc
        self.save_failed_episodes = save_failed_episodes  # 실패한 에피소드 저장 여부
        self.max_steps_per_episode = max_steps_per_episode  # 에피소드당 최대 스텝 수 제한
        self.target_uoc = self._extract_uoc_from_path(model_path)
        global ENV
        self.env = ENV
        self.model = SAC.load(model_path)
        
        self.results = {
            'uoc': [],
            'success_rate': [],
            'avg_episode_length': [],
            'avg_distance': [],
            'min_distance': [],
            'trajectory_efficiency': []
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.demonstration_data = {i: [] for i in range(1, 9)}  # UoC별 데이터 저장
    
    def _extract_uoc_from_path(self, model_path):
        # 모델 파일명에서 UoC 번호 추출
        filename = os.path.basename(model_path)
        for i in range(1, 9):
            if f'UoC_{i}' in filename:
                return i
        return None
        
    def validate_uoc(self, uoc):
        global DETERMINISTIC
        episode_lengths = []
        successes = []
        distances = []
        trajectory_lengths = []

        self.env.Curriculum_manager.current_uoc = uoc
        
        # Target XYZ 목록을 순차적으로 읽어오기
        target_data = pd.read_csv(f"./Preprocessing_datas/8000points_curriculum/data/UoC_{uoc}.csv")
        
        for target_xyz in tqdm(target_data.values, desc=f'Validating UoC {self.env.Curriculum_manager.current_uoc}'):
            # 타겟 설정
            self.env.target = target_xyz.tolist()
            self.env.Curriculum_manager.target = target_xyz.tolist()
            
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_length = 0
            min_distance = float('inf')
            trajectory_length = 0
            last_ee_pos = None
            episode_data = []
            
            while not (done or truncated):
                state = obs.copy()  # 현재 상태 저장
                action, _ = self.model.predict(obs, deterministic=DETERMINISTIC)
                next_obs, _, done, truncated, info = self.env.step(action)
                
                episode_length += 1
                min_distance = min(min_distance, self.env.distance)
                
                # 성공한 에피소드의 경우 데이터 저장
                if not truncated:
                    episode_data.append({
                        'state': state,
                        'action': action,
                        'next_state': next_obs,
                        'reward': self.env.last_reward if hasattr(self.env, 'last_reward') else 0.0
                    })
                
                obs = next_obs
            
            # 저장 조건 체크: 성공/실패 여부와 최대 스텝 수 제한 확인
            should_save = (self.save_failed_episodes or info.get('is_success', False))
            if self.max_steps_per_episode is not None:
                should_save = should_save and (episode_length < self.max_steps_per_episode)
                
            if should_save:
                self.demonstration_data[uoc].extend(episode_data)
            
            episode_lengths.append(episode_length)
            successes.append(info.get('is_success', False))
            distances.append(min_distance)
            trajectory_lengths.append(trajectory_length)
        
        # 결과 계산 및 저장 - Add safe calculations for empty arrays
        success_rate = np.mean(successes) * 100 if successes else 0.0
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_distance = np.mean(distances) if distances else float('inf')
        min_distance_achieved = np.min(distances) if distances else float('inf')
        
        # Safe calculation of trajectory efficiency
        avg_trajectory_efficiency = 0.0
        if trajectory_lengths:
            valid_efficiencies = [d/t for d, t in zip(distances, trajectory_lengths) if t > 0]
            avg_trajectory_efficiency = np.mean(valid_efficiencies) if valid_efficiencies else 0.0
        
        self.results['uoc'].append(uoc)
        self.results['success_rate'].append(success_rate)
        self.results['avg_episode_length'].append(avg_episode_length)
        self.results['avg_distance'].append(avg_distance)
        self.results['min_distance'].append(min_distance_achieved)
        self.results['trajectory_efficiency'].append(avg_trajectory_efficiency)
        
        # UoC별 데이터를 CSV 파일로 저장
        if self.demonstration_data[uoc]:  # 데이터가 있는 경우에만 저장
            df_data = []
            for entry in self.demonstration_data[uoc]:
                df_data.append({
                    'state': ','.join(map(str, entry['state'])),
                    'action': ','.join(map(str, entry['action'])),
                    'next_state': ','.join(map(str, entry['next_state'])),
                    'reward': entry['reward']
                })
            
            df = pd.DataFrame(df_data)
            output_path = os.path.join(self.output_dir, f'UoC_{uoc}.csv')
            
            # 파일이 존재하면 헤더 없이 추가, 없으면 헤더와 함께 새로 생성
            if os.path.exists(output_path):
                df.to_csv(output_path, mode='a', index=False, header=False)
            else:
                df.to_csv(output_path, mode='w', index=False, header=True)
                
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_episode_length,
            'avg_distance': avg_distance,
            'min_distance': min_distance_achieved,
            'trajectory_efficiency': avg_trajectory_efficiency
        }
    
    def validate_all_uocs(self):
        if self.target_uoc is not None:
            # 특정 UoC만 검증
            self.validate_uoc(self.target_uoc)
        else:
            # UoC가 지정되지 않은 경우 모든 UoC 검증
            max_uoc = self.env.Curriculum_manager.max_uoc
            for uoc in range(1, max_uoc + 1):
                self.validate_uoc(uoc)
        
        # 모든 UoC 데이터를 하나의 DataFrame으로 통합
        all_data = []
        for uoc in range(1, 9):
            if self.demonstration_data[uoc]:
                for entry in self.demonstration_data[uoc]:
                    data_entry = {
                        'uoc': uoc,
                        'state': ','.join(map(str, entry['state'])),
                        'action': ','.join(map(str, entry['action'])),
                        'next_state': ','.join(map(str, entry['next_state'])),
                        'reward': entry['reward']
                    }
                    all_data.append(data_entry)
        
        if all_data:
            df_all_data = pd.DataFrame(all_data)
            output_path = os.path.join(self.output_dir, 'all_uocs_data.csv')
            
            # 통합 파일도 append 모드로 저장
            if os.path.exists(output_path):
                df_all_data.to_csv(output_path, mode='a', index=False, header=False)
            else:
                df_all_data.to_csv(output_path, mode='w', index=False, header=True)
            
        return pd.DataFrame(self.results)

def validate_all_models(base_dir='models', save_failed_episodes=False, max_steps_per_episode=None, output_dir="demonstration_inform"):
    exp_folders = glob.glob(os.path.join(base_dir, 'Ex(*)/'))
    
    # 결과 저장을 위한 디렉토리 생성
    validation_dir = os.path.join(output_dir, 'validation_results')
    os.makedirs(validation_dir, exist_ok=True)
    
    for exp_folder in exp_folders:
        model_files = glob.glob(os.path.join(exp_folder, '*.zip'))
        
        for model_path in model_files:
            print(f"\nValidating model: {model_path}")
            
            try:
                # demonstration_inform 디렉토리는 output_dir 아래에 생성
                demonstration_dir = os.path.join(output_dir, 'demonstration_inform')
                os.makedirs(demonstration_dir, exist_ok=True)
                
                validator = ModelValidator(
                    model_path, 
                    save_failed_episodes=save_failed_episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    output_dir=demonstration_dir
                )
                results_df = validator.validate_all_uocs()
                
                results_df['experiment'] = os.path.basename(os.path.dirname(model_path))
                results_df['model_name'] = os.path.basename(model_path)
                validation_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_df['validation_timestamp'] = validation_timestamp
                
                # validation 결과를 구성별 폴더에 저장
                results_path = os.path.join(
                    validation_dir,
                    f'results_deterministic_{DETERMINISTIC}.csv'
                )
                file_exists = os.path.isfile(results_path)
                results_df.to_csv(results_path, mode='a', index=False, header=not file_exists)
                
            except Exception as e:
                print(f"Error validating {model_path}: {str(e)}")
                continue
    
    return None

if __name__ == "__main__":
    # 8가지 설정 조합 정의
    configurations = [
        {"deterministic": True, "save_failed": True, "max_steps": 20, "dir": "demonstration_det_true_failed_true_steps_20"},
        {"deterministic": True, "save_failed": True, "max_steps": 100, "dir": "demonstration_det_true_failed_true_steps_100"},
        {"deterministic": True, "save_failed": False, "max_steps": 20, "dir": "demonstration_det_true_failed_false_steps_20"},
        {"deterministic": True, "save_failed": False, "max_steps": 100, "dir": "demonstration_det_true_failed_false_steps_100"},
        {"deterministic": False, "save_failed": True, "max_steps": 20, "dir": "demonstration_det_false_failed_true_steps_20"},
        {"deterministic": False, "save_failed": True, "max_steps": 100, "dir": "demonstration_det_false_failed_true_steps_100"},
        {"deterministic": False, "save_failed": False, "max_steps": 20, "dir": "demonstration_det_false_failed_false_steps_20"},
        {"deterministic": False, "save_failed": False, "max_steps": 100, "dir": "demonstration_det_false_failed_false_steps_100"},
    ]

    # 각 설정에 대해 순차적으로 실행
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"Running with configuration:")
        print(f"Deterministic: {config['deterministic']}")
        print(f"Save Failed Episodes: {config['save_failed']}")
        print(f"Max Steps: {config['max_steps']}")
        print(f"Output Directory: {config['dir']}")
        print(f"{'='*50}\n")

        DETERMINISTIC = config['deterministic']
        
        # 각 설정별 base 디렉토리 생성
        os.makedirs(config['dir'], exist_ok=True)
        
        results = validate_all_models(
            save_failed_episodes=config['save_failed'],
            max_steps_per_episode=config['max_steps'],
            output_dir=config['dir']
        )

        if results is not None:
            print("\nValidation Results Summary:")
            print(results.to_string(index=False))
            
        print(f"\nCompleted configuration: {config['dir']}\n")