import os
import gc
import time
import torch
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import wandb
import numpy as np
from datetime import datetime

from utils import get_random_name
from GenesisEnv import Genesis_Simulator

class CustomLoggingCallback(BaseCallback):
    def __init__(
        self, 
        save_freq: int, 
        save_path: str, 
        prefix_name: str, 
        wandb_run=None, 
        verbose=0
    ):
        super().__init__(verbose)
        # Model saving settings
        self.save_freq = save_freq
        self.save_path = save_path
        self.prefix_name = prefix_name
        self.run = wandb_run
        
        # Episode tracking
        self.episode_count = 0
        self.training_start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        # Success rate tracking
        self.success_window = 200 #trick
        self.success_history = [False * self.success_window]
        
        # Loss tracking
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.last_ent_coef = None
        

        # CSV logging setup
        self.log_path = os.path.join(self.save_path, "logs")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        # CSV file paths setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_csv_path = os.path.join(self.log_path, f"metrics_{timestamp}.csv")
        
        # Write CSV headers
        self.write_csv_headers()

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _save_model(self):
        """Save the model and optionally upload to wandb"""
        try:
            # Create filename with episode number
            filename = f"{self.prefix_name}_episode_{self.episode_count}.zip"
            path = os.path.join(self.save_path, filename)
            
            # Save the model
            self.model.save(path)
            
            if self.verbose > 0:
                print(f"Saved model to {path}")
            
            # If wandb is enabled, log the model as an artifact
            if self.run is not None:
                artifact = wandb.Artifact(
                    name=f"{self.prefix_name}_episode_{self.episode_count}",
                    type="model"
                )
                artifact.add_file(path)
                self.run.log_artifact(artifact)
                
                if self.verbose > 0:
                    print(f"Uploaded model to wandb as artifact")
                    
        except Exception as e:
            print(f"Error saving model: {e}")
            
    def write_csv_headers(self):
        """CSV 파일 헤더 작성"""
        # 메트릭스 CSV 헤더
        metrics_headers = [
            "timestamp",
            "episode",
            "total_timesteps",
            "fps",
            "episode_reward",
            "episode_length",
            "episode_reward_mean",
            "episode_length_mean",
            "success_rate",
            "replay_buffer_size",
            "replay_buffer_usage",
            "actor_loss",
            "critic_loss",
            "ent_coef"
        ]

        # CSV 파일 생성 및 헤더 작성
        pd.DataFrame(columns=metrics_headers).to_csv(self.metrics_csv_path, index=False)

    def log_to_csv(self, metrics):
        """메트릭스를 CSV 파일에 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 메트릭스 데이터 준비
        metrics_data = {
            "timestamp": timestamp,
            "episode": self.episode_count,
            "total_timesteps": metrics.get("time/total_timesteps", 0),
            "fps": metrics.get("time/fps", 0),
            "episode_reward": metrics.get("rollout/episode_reward", 0),
            "episode_length": metrics.get("rollout/episode_length", 0),
            "episode_reward_mean": metrics.get("rollout/episode_reward_mean", 0),
            "episode_length_mean": metrics.get("rollout/episode_length_mean", 0),
            "success_rate": metrics.get("rollout/success_rate", 0),
            "replay_buffer_size": metrics.get("replay_buffer/size", 0),
            "replay_buffer_usage": metrics.get("replay_buffer/usage_percent", 0),
            "actor_loss": metrics.get("train/actor_loss", None),
            "critic_loss": metrics.get("train/critic_loss", None),
            "ent_coef": metrics.get("train/ent_coef", None)
        }
        
        # 메트릭스 CSV에 추가
        pd.DataFrame([metrics_data]).to_csv(self.metrics_csv_path, mode='a', header=False, index=False)
        
    def _on_training_start(self):
        self.training_start_time = time.time()

    def _update_success_rate(self, is_success: bool):
        """성공률 업데이트"""
        # 전체 성공률 업데이트
        self.success_history.append(int(is_success))
        if len(self.success_history) > self.success_window:
            self.success_history.pop(0)
            
    def _get_success_rates(self):
        # 전체 성공률
        overall_rate = np.mean(self.success_history) * 100 if self.success_history else 0
        return overall_rate

    def _on_step(self) -> bool:
        # 현재 스텝의 보상과 길이 누적
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_steps += 1

        # Loss 값 추적
        try:
            sac_model = self.locals["self"]
            if hasattr(sac_model, '_n_updates') and sac_model._n_updates > 0:
                if hasattr(sac_model, 'logger') and hasattr(sac_model.logger, 'name_to_value'):
                    logs = sac_model.logger.name_to_value
                    if 'train/actor_loss' in logs:
                        self.last_actor_loss = logs['train/actor_loss']
                    if 'train/critic_loss' in logs:
                        self.last_critic_loss = logs['train/critic_loss']
                    if 'train/ent_coef' in logs:
                        self.last_ent_coef = logs['train/ent_coef']
        except Exception as e:
            print(f"Error tracking losses: {e}")

        # info 딕셔너리에서 정보 추출
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:
                self.episode_count += 1

                # Success 정보 업데이트
                is_success = info.get("is_success", False)
                self._update_success_rate(is_success)
                
                # Replay buffer 정보
                replay_buffer = self.locals["replay_buffer"]
                current_size = replay_buffer.size()
                max_size = replay_buffer.buffer_size
                memory_usage = current_size / max_size * 100
                
                # 학습 시간 계산
                training_time = time.time() - self.training_start_time
                
                # 에피소드 메트릭 저장
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_steps)
                recent_mean_reward = np.mean(self.episode_rewards[-100:])
                
                # 성공률 계산
                overall_success_rate = self._get_success_rates()
                
                # wandb에 로깅할 메트릭스 준비
                metrics = {
                    "rollout/episode_reward": [self.episode_count, self.current_episode_reward],
                    "rollout/episode_length": [self.episode_count, self.current_episode_steps],
                    "rollout/episode_reward_mean": [self.episode_count, recent_mean_reward],
                    "rollout/episode_length_mean": [self.episode_count, np.mean(self.episode_lengths[-100:])],
                    "rollout/success_rate": [self.episode_count, overall_success_rate],
                    "time/total_timesteps": [self.episode_count, self.num_timesteps],
                    "time/fps": [self.episode_count, int(self.num_timesteps / (training_time + 1e-8))],
                    "replay_buffer/size": [self.episode_count, current_size],
                    "replay_buffer/usage_percent": [self.episode_count, memory_usage],
                }

                # Loss 메트릭 추가
                if self.last_actor_loss is not None:
                    metrics["train/actor_loss"] = [self.episode_count, self.last_actor_loss]
                if self.last_critic_loss is not None:
                    metrics["train/critic_loss"] = [self.episode_count, self.last_critic_loss]
                if self.last_ent_coef is not None:
                    metrics["train/ent_coef"] = [self.episode_count, self.last_ent_coef]

                # CSV 파일에 로깅을 위한 메트릭스
                csv_metrics = {k: v[1] for k, v in metrics.items()}  # CSV에는 값만 저장

                # CSV 파일에 로깅
                self.log_to_csv(csv_metrics)

                # wandb에 로깅
                if self.run is not None:
                    # 기본 메트릭스 로깅
                    for metric_name, (step, value) in metrics.items():
                        wandb.log({metric_name: value}, step=step)
                
                # 주기적인 모델 저장
                if self.episode_count % self.save_freq == 0:
                    self._save_model()
                
                if overall_success_rate >= 100:
                    raise Exception("Done")
                
                # 다음 에피소드를 위한 초기화
                self.current_episode_reward = 0
                self.current_episode_steps = 0

        return True

def train_genesis(
    algorithm="SAC",
    total_timesteps=30_000_000,
    seed=42,
    num_envs=1,
    learning_rate=0.0005,
    batch_size=1024,
    n_epochs=10,
    gamma=0.9,
    buffer_size=30_000_000,
    learning_starts=2048,
    train_freq=10,
    gradient_steps=8,
    save_freq=1000,
    device="auto",
    random_name=None,
    UoC_dir = None,
    UoC_name = None,
    env = None
):
    UoC_path = os.path.join(UoC_dir, f"{UoC_name}.csv")
    if (env == None):
        env = Genesis_Simulator(render=False, UoC_path=UoC_path)
        env = Monitor(env)
    else:
        env.env.UoC_path=UoC_path
        env.env._initialize_rl_parameters()
        env.env.generate_target_csv()

    """Genesis Simulator 훈련 함수"""
    if random_name is None:
        random_name = get_random_name(UoC_name)

    # 설정 저장
    config = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "num_envs": num_envs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "net_arch": [64, 64],
        "device": device
    }

    # wandb 초기화
    run = wandb.init(
        project="genesis-robot-each-UoCs",
        name=random_name,
        config=config,
        monitor_gym=True,
        save_code=True,
    )
    
    # 실험 결과 저장 디렉토리 생성
    save_path = os.path.join("training_results", random_name)
    os.makedirs(save_path, exist_ok=True)

    if algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs={"net_arch": [64, 64]},
            device=device,
            verbose=2,  # 기본 로깅 비활성화
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 커스텀 콜백 설정
    callbacks = [
        CustomLoggingCallback(
            save_freq=save_freq,
            save_path=os.path.join(save_path, "models"),
            prefix_name=random_name,
            wandb_run=run,
            verbose=1
        )
    ]

    # 학습 시작
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
    
    finally:
        # 최종 모델 저장
        final_model_path = os.path.join(save_path, f"{random_name}_final.zip")
        model.save(final_model_path)
        
        # 최종 모델을 W&B에 업로드
        artifact = wandb.Artifact(name=f"{random_name}_{gamma}_{learning_rate}_final", type="model")
        artifact.add_file(final_model_path)
        run.log_artifact(artifact)

        run.finish()
    # env.close()
        
    return model, env



if __name__ == "__main__":

    env = None
    learning_UoCs = [1,2,3,4]

    for learning_UoC in learning_UoCs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # SAC 알고리즘으로 학습 실행
        config = {
            "algorithm": "SAC",
            "total_timesteps": 30_000_000,
            "seed": 42,
            "num_envs": 1,
            "learning_rate": 0.0003,
            "batch_size": 1024,
            "gamma": 0.90,
            "buffer_size": 30_000_000,
            "learning_starts": 2048,
            "train_freq": 10,
            "gradient_steps": 8,
            "device": device,
            "UoC_dir" : "Preprocessing_datas/2025-01-20_22-11-34/data/",
            "UoC_name" : f"UoC_{learning_UoC}",
            "env" : env
        }
        
        # 모델 학습
        model, env = train_genesis(**config)
        env = env
        gc.collect()
