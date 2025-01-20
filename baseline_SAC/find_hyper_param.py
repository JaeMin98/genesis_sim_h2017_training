import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from collections import defaultdict
from tqdm import tqdm
import torch

import train

class HyperParameterRanges:
    """하이퍼파라미터 범위 정의 클래스"""
    def __init__(self, 
                 # 공통 파라미터 범위
                 lr_range=(1e-6, 1e-2),
                 batch_size_options=[256, 512, 1024, 2048, 4096, 8192],
                 gamma_range=(0.8, 0.9999),
                 total_timesteps=30000,
                 # PPO 전용 파라미터 범위
                 ppo_epochs_range=(3, 30),
                 ppo_n_steps_options=[64, 128, 256, 512, 1024, 2048],
                 ppo_clip_range=(0.1, 0.4),
                 ppo_ent_coef_range=(1e-8, 1e-1),
                 ppo_vf_coef_range=(0.1, 0.9),
                 ppo_max_grad_norm_range=(0.1, 1.0),
                 ppo_target_kl_range=(0.001, 0.1),
                 # SAC 전용 파라미터 범위
                 sac_buffer_size_options=[10_000, 50_000, 100_000, 500_000, 1_000_000],
                 sac_learning_starts_range=(100, 10000),
                 sac_train_freq_range=(1, 64),
                 sac_gradient_steps_range=(1, 64),
                 sac_tau_range=(0.001, 0.2),
                 sac_target_entropy_range=(-10, 0),
                 # 네트워크 구조 옵션
                 net_arch_options=[[64, 64], [128, 128], [256, 256], [400, 300],
                                 [64, 32], [32, 32], [64, 64, 64]]):
        # 공통 파라미터
        self.lr_range = lr_range
        self.batch_size_options = batch_size_options
        self.gamma_range = gamma_range
        self.total_timesteps = total_timesteps
        
        # PPO 전용 파라미터
        self.ppo_epochs_range = ppo_epochs_range
        self.ppo_n_steps_options = ppo_n_steps_options
        self.ppo_clip_range = ppo_clip_range
        self.ppo_ent_coef_range = ppo_ent_coef_range
        self.ppo_vf_coef_range = ppo_vf_coef_range
        self.ppo_max_grad_norm_range = ppo_max_grad_norm_range
        self.ppo_target_kl_range = ppo_target_kl_range
        
        # SAC 전용 파라미터
        self.sac_buffer_size_options = sac_buffer_size_options
        self.sac_learning_starts_range = sac_learning_starts_range
        self.sac_train_freq_range = sac_train_freq_range
        self.sac_gradient_steps_range = sac_gradient_steps_range
        self.sac_tau_range = sac_tau_range
        self.sac_target_entropy_range = sac_target_entropy_range
        
        # 네트워크 구조
        self.net_arch_options = net_arch_options

class ExperimentTracker:
    """실험 결과를 추적하고 저장하는 클래스"""
    def __init__(self, base_dir: str = "optimization_results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(base_dir, f"optimization_{self.timestamp}")
        self.results = []
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
    def add_trial_result(self, trial_number: int, hyperparameters: Dict, metrics: Dict):
        """각 trial의 결과 추가"""
        result = {
            "trial_number": trial_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **hyperparameters,
            **metrics
        }
        
        self.results.append(result)
        self.save_results()
        
    def save_results(self):
        """결과를 CSV 파일로 저장"""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(self.results_dir, "optimization_results.csv")
            df.to_csv(csv_path, index=False)
        
    def save_best_trial(self, best_trial: Dict):
        """최적의 trial 결과를 별도 파일로 저장"""
        best_params_path = os.path.join(self.results_dir, "best_parameters.json")
        with open(best_params_path, 'w') as f:
            json.dump(best_trial, f, indent=4)
            
    def get_results_dir(self) -> str:
        """결과 디렉토리 경로 반환"""
        return self.results_dir

def evaluate_model(model, env, n_eval_episodes: int = 10) -> Dict[str, float]:
    """모델 성능 평가"""
    rewards = []
    episode_lengths = []
    success_rate = 0
    

    for episode in tqdm(range(n_eval_episodes), desc="Evaluating model", leave=False):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if done:
                rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if info.get("is_success", False):
                    success_rate += 1
                break
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "success_rate": success_rate / n_eval_episodes
    }

def get_algorithm_params(trial: optuna.Trial, algorithm: str, param_ranges: HyperParameterRanges) -> Dict[str, Any]:
    """알고리즘별 하이퍼파라미터 생성"""
    # train_genesis가 직접 받는 파라미터만 포함
    params = {
        "algorithm": algorithm,
        "total_timesteps": param_ranges.total_timesteps,
        "seed": 42,
        "num_envs": 1,  # Genesis는 1개만 가능
        "learning_rate": trial.suggest_loguniform("learning_rate", *param_ranges.lr_range),
        "batch_size": trial.suggest_categorical("batch_size", param_ranges.batch_size_options),
        "gamma": trial.suggest_uniform("gamma", *param_ranges.gamma_range),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # 알고리즘별 파라미터
    if algorithm == "PPO":
        params.update({
            "n_epochs": trial.suggest_int("n_epochs", *param_ranges.ppo_epochs_range),
        })
    else:  # SAC
        params.update({
            "buffer_size": trial.suggest_categorical("buffer_size", param_ranges.sac_buffer_size_options),
            "learning_starts": trial.suggest_int("learning_starts", *param_ranges.sac_learning_starts_range),
            "train_freq": trial.suggest_int("train_freq", *param_ranges.sac_train_freq_range),
            "gradient_steps": trial.suggest_int("gradient_steps", *param_ranges.sac_gradient_steps_range),
        })
    
    return params

def objective(trial: optuna.Trial, tracker: ExperimentTracker, param_ranges: HyperParameterRanges) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # 알고리즘 선택
    algorithm = trial.suggest_categorical("algorithm", ["PPO", "SAC"])
    
    # 알고리즘별 하이퍼파라미터 설정
    hyperparameters = get_algorithm_params(trial, algorithm, param_ranges)

    try:
        # 모델 학습
        start_time = datetime.now()
        model, env = train.train_genesis(**hyperparameters)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 모델 평가
        eval_metrics = evaluate_model(model, env)
        
        # 모든 메트릭 결합
        metrics = {
            "training_time": training_time,
            "status": "completed",
            **eval_metrics
        }
        
        # 결과 저장
        tracker.add_trial_result(
            trial_number=trial.number,
            hyperparameters=hyperparameters,
            metrics=metrics
        )
        
        print(f"\nTrial {trial.number} completed:")
        print(f"Algorithm: {algorithm}")
        print(f"Mean reward: {eval_metrics['mean_reward']:.2f}")
        print(f"Success rate: {eval_metrics['success_rate']:.2%}")
        print(f"Training time: {training_time:.2f}s")
        
        return eval_metrics["mean_reward"]

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # 실패한 시도도 기록
        tracker.add_trial_result(
            trial_number=trial.number,
            hyperparameters=hyperparameters,
            metrics={
                "status": "failed",
                "error": str(e),
                "training_time": 0,
                "mean_reward": float('-inf'),
                "std_reward": 0,
                "min_reward": 0,
                "max_reward": 0,
                "mean_episode_length": 0,
                "success_rate": 0
            }
        )
        raise optuna.exceptions.TrialPruned()



def optimize_hyperparameters(
    param_ranges: HyperParameterRanges,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = None,
    storage: str = None,
) -> Dict[str, Any]:
    """하이퍼파라미터 최적화 실행 함수"""
    
    if study_name is None:
        study_name = f"genesis-optimization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 결과 트래커 초기화
    tracker = ExperimentTracker()
    
    # Optuna 설정
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        n_min_trials=10
    )
    
    # Study 생성 및 최적화 실행
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    
    # 진행 상황 표시를 위한 progress bar
    pbar = tqdm(total=n_trials, desc="Optimization Progress")
    
    def objective_with_progress(trial):
        value = objective(trial, tracker, param_ranges)
        pbar.update(1)
        return value
    
    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials,
            n_jobs=n_jobs
        )
    finally:
        pbar.close()
    
    # 최적의 하이퍼파라미터 저장
    best_trial_info = {
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tracker.save_best_trial(best_trial_info)
    
    # 결과 요약
    print("\nOptimization Results Summary:")
    print("=" * 50)
    print(f"Results saved in: {tracker.get_results_dir()}")
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params

if __name__ == "__main__":
    print("\n=== Genesis Robot Hyperparameter Optimization ===")
    print("Starting optimization process...")
    
    # 하이퍼파라미터 범위 설정
    param_ranges = HyperParameterRanges(
        # 공통 파라미터
        lr_range=(1e-5, 1e-3),                    # 학습률 범위
        batch_size_options=[1024, 2048, 4096],    # 배치 크기 옵션
        gamma_range=(0.95, 0.995),                # 감마값 범위
        total_timesteps=100000,                    # 1회당 타임스텝
        
        # PPO 전용 파라미터
        ppo_epochs_range=(5, 15),                 # PPO epoch 범위
        ppo_n_steps_options=[128, 256, 512],      # PPO 스텝 옵션
        ppo_clip_range=(0.1, 0.3),               # PPO 클리핑 범위
        ppo_ent_coef_range=(1e-6, 1e-3),         # 엔트로피 계수 범위
        ppo_vf_coef_range=(0.5, 0.9),            # 가치 함수 계수 범위
        ppo_max_grad_norm_range=(0.3, 0.7),      # 그래디언트 클리핑 범위
        ppo_target_kl_range=(0.01, 0.05),        # KL 발산 목표값 범위
        
        # SAC 전용 파라미터
        sac_buffer_size_options=[50_000, 100_000, 500_000],  # 버퍼 크기 옵션
        sac_learning_starts_range=(1000, 5000),             # 학습 시작 타임스텝 범위
        sac_train_freq_range=(1, 32),                       # 학습 빈도 범위
        sac_gradient_steps_range=(1, 32),                   # 그래디언트 스텝 범위
        sac_tau_range=(0.001, 0.1),                        # TAU 범위
        sac_target_entropy_range=(-5, -1),                 # 목표 엔트로피 범위
        
        # 네트워크 구조 옵션
        net_arch_options=[
            [64, 64],
            [128, 128],
            [256, 256],
            [64, 32],
            [32, 32]
        ]
    )
    
    try:
        # 최적화 실행
        best_params = optimize_hyperparameters(
            param_ranges=param_ranges,
            n_trials=50,           # 총 시도할 횟수
            n_jobs=1,             # 병렬 처리 수 (Genesis는 1개만 가능)
            study_name="genesis_optimization"  # 스터디 이름
        )
        
        print("\n=== Training Final Model ===")
        print("Best parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
            
        # 최적의 하이퍼파라미터로 최종 모델 학습
        final_model, final_env = train.train_genesis(**best_params)
        
        # 최종 모델 평가
        print("\n=== Evaluating Final Model ===")
        final_metrics = evaluate_model(final_model, final_env, n_eval_episodes=20)
        
        print("\nFinal Model Performance:")
        print(f"Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
        print(f"Success Rate: {final_metrics['success_rate']:.2%}")
        print(f"Average Episode Length: {final_metrics['mean_episode_length']:.1f}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
    finally:
        print("\nOptimization process completed")