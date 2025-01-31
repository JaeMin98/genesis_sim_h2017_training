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
DETERMINISTIC = True
NUM_OF_EPISODE = 100

class ModelValidator:
    def __init__(self, model_path, num_episodes_per_uoc=100):
        self.model_path = model_path
        self.num_episodes_per_uoc = num_episodes_per_uoc
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
        
    def validate_uoc(self, uoc):
        global DETERMINISTIC
        episode_lengths = []
        successes = []
        distances = []
        trajectory_lengths = []

        self.env.Curriculum_manager.current_uoc = uoc
        
        for _ in tqdm(range(self.num_episodes_per_uoc), desc=f'Validating UoC {self.env.Curriculum_manager.current_uoc}'):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_length = 0
            min_distance = float('inf')
            trajectory_length = 0
            last_ee_pos = None
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=DETERMINISTIC)
                obs, _, done, truncated, info = self.env.step(action)
                
                episode_length += 1
                min_distance = min(min_distance, self.env.distance)
                
                ee_pos = self.env.H2017.get_link("link6").get_pos()
                if hasattr(ee_pos, 'tolist'):
                    ee_pos = ee_pos.tolist()
                
                if last_ee_pos is not None:
                    trajectory_length += np.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, last_ee_pos)))
                last_ee_pos = ee_pos
            
            episode_lengths.append(episode_length)
            successes.append(info.get('is_success', False))
            distances.append(min_distance)
            trajectory_lengths.append(trajectory_length)
        
        success_rate = np.mean(successes) * 100
        avg_episode_length = np.mean(episode_lengths)
        avg_distance = np.mean(distances)
        min_distance_achieved = np.min(distances)
        
        avg_trajectory_efficiency = np.mean([d/t if t > 0 else 0 
                                          for d, t in zip(distances, trajectory_lengths)])
        
        self.results['uoc'].append(uoc)
        self.results['success_rate'].append(success_rate)
        self.results['avg_episode_length'].append(avg_episode_length)
        self.results['avg_distance'].append(avg_distance)
        self.results['min_distance'].append(min_distance_achieved)
        self.results['trajectory_efficiency'].append(avg_trajectory_efficiency)
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_episode_length,
            'avg_distance': avg_distance,
            'min_distance': min_distance_achieved,
            'trajectory_efficiency': avg_trajectory_efficiency
        }
    
    def validate_all_uocs(self):
        max_uoc = self.env.Curriculum_manager.max_uoc
        
        for uoc in range(1, max_uoc + 1):
            self.validate_uoc(uoc)
            
        return pd.DataFrame(self.results)

def validate_all_models(base_dir='models', num_episodes_per_uoc=100):
    exp_folders = glob.glob(os.path.join(base_dir, 'Ex(*)/'))
    
    for exp_folder in exp_folders:
        model_files = glob.glob(os.path.join(exp_folder, '*.zip'))
        
        for model_path in model_files:
            print(f"\nValidating model: {model_path}")
            
            try:
                validator = ModelValidator(model_path, num_episodes_per_uoc)
                results_df = validator.validate_all_uocs()
                
                results_df['experiment'] = os.path.basename(os.path.dirname(model_path))
                results_df['model_name'] = os.path.basename(model_path)
                validation_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_df['validation_timestamp'] = validation_timestamp
                
                output_dir = 'validation_results'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f'results_deterministic_{DETERMINISTIC}_episode_{NUM_OF_EPISODE}.csv'
                )
                file_exists = os.path.isfile(output_path)
                results_df.to_csv(output_path, mode='a', index=False, header=not file_exists)
                
            except Exception as e:
                print(f"Error validating {model_path}: {str(e)}")
                continue
    
    return None

if __name__ == "__main__":
    DETERMINISTIC = True
    results = validate_all_models(num_episodes_per_uoc=NUM_OF_EPISODE)
    
    if results is not None:
        print("\nCombined Validation Results Summary:")
        print(results.to_string(index=False))

    DETERMINISTIC = False
    results = validate_all_models(num_episodes_per_uoc=NUM_OF_EPISODE)
    
    if results is not None:
        print("\nCombined Validation Results Summary:")
        print(results.to_string(index=False))