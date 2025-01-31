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
    def __init__(self, model_path, num_episodes_per_Region=100):
        self.model_path = model_path
        self.num_episodes_per_Region = num_episodes_per_Region
        global ENV
        self.env = ENV
        self.model = SAC.load(model_path)
        
        self.results = {
            'Region': [],
            'success_rate': [],
            'avg_episode_length': [],
            'avg_distance': [],
            'min_distance': [],
            'trajectory_efficiency': []
        }
        
    def validate_Region(self, Region):
        global DETERMINISTIC
        episode_lengths = []
        successes = []
        distances = []
        trajectory_lengths = []

        self.env.Curriculum_manager.current_Region = Region
        
        for _ in tqdm(range(self.num_episodes_per_Region), desc=f'Validating Region {self.env.Curriculum_manager.current_Region}'):
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
        
        self.results['Region'].append(Region)
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
    
    def validate_all_Regions(self):
        max_Region = self.env.Curriculum_manager.max_Region
        
        for Region in range(1, max_Region + 1):
            self.validate_Region(Region)
            
        return pd.DataFrame(self.results)
    
    def save_results(self, output_dir='validation_results'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f'{output_dir}/validation_results_{timestamp}.csv', index=False)
        
        self._create_visualizations(output_dir, timestamp)
        
    def _create_visualizations(self, output_dir, timestamp):
        metrics = ['success_rate', 'avg_episode_length', 'avg_distance', 'trajectory_efficiency']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            axes[idx].bar(self.results['Region'], self.results[metric])
            axes[idx].set_title(f'{metric.replace("_", " ").title()} by Region')
            axes[idx].set_xlabel('Region')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/validation_metrics_{timestamp}.png')
        plt.close()

def validate_all_models(base_dir='models', num_episodes_per_Region=100):
    all_results = []
    exp_folders = glob.glob(os.path.join(base_dir, 'Ex(*)/'))
    
    for exp_folder in exp_folders:
        model_files = glob.glob(os.path.join(exp_folder, '*.zip'))
        
        for model_path in model_files:
            print(f"\nValidating model: {model_path}")
            
            try:
                validator = ModelValidator(model_path, num_episodes_per_Region)
                results_df = validator.validate_all_Regions()
                
                results_df['experiment'] = os.path.basename(os.path.dirname(model_path))
                results_df['model_name'] = os.path.basename(model_path)
                
                all_results.append(results_df)
                
            except Exception as e:
                print(f"Error validating {model_path}: {str(e)}")
                continue
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'validation_results'
        os.makedirs(output_dir, exist_ok=True)
        global DETERMINISTIC, NUM_OF_EPISODE
        output_path = os.path.join(output_dir, f'results_deterministic_{DETERMINISTIC}_episode_{NUM_OF_EPISODE}_{timestamp}.csv')
        combined_results.to_csv(output_path, index=False)
        
        print(f"\nAll validation results saved to: {output_path}")
        return combined_results
    else:
        print("No validation results were generated.")
        return None

if __name__ == "__main__":
    DETERMINISTIC = True
    results = validate_all_models(num_episodes_per_Region=NUM_OF_EPISODE)
    
    if results is not None:
        print("\nCombined Validation Results Summary:")
        print(results.to_string(index=False))

    DETERMINISTIC = False
    results = validate_all_models(num_episodes_per_Region=NUM_OF_EPISODE)
    
    if results is not None:
        print("\nCombined Validation Results Summary:")
        print(results.to_string(index=False))