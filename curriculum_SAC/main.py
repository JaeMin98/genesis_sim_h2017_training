import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from utils import get_random_name
from GenesisEnv import Genesis_Simulator


class SaveModelEveryNEpisodesCallback(BaseCallback):
    """지정된 에피소드마다 모델을 저장하고 W&B에 업로드하는 콜백"""
    def __init__(self, n_episodes: int, save_path: str, prefix_name: str, wandb_run=None, verbose=0):
        super(SaveModelEveryNEpisodesCallback, self).__init__(verbose)
        self.n_episodes = n_episodes
        self.save_path = save_path
        self.prefix_name = prefix_name
        self.run = wandb_run
        self.episode_count = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                if self.episode_count % self.n_episodes == 0:
                    # 로컬에 모델 저장
                    filename = os.path.join(
                        self.save_path,
                        f"{self.prefix_name}_ep{self.episode_count}.tar"
                    )
                    self.model.save(filename)
                    if self.verbose > 0:
                        print(f"[Callback] Model saved to {filename}")

                    # W&B에 모델 업로드
                    if self.run is not None:
                        artifact_name = f"{self.prefix_name}_ep{self.episode_count}"
                        artifact = wandb.Artifact(name=artifact_name, type="model")
                        artifact.add_file(filename)
                        self.run.log_artifact(artifact)
                        if self.verbose > 0:
                            print(f"[Callback] Model artifact uploaded to W&B: {artifact_name}")
        return True

def make_env(rank, seed=0):
    """환경 생성 헬퍼 함수"""
    def _init():
        env = Genesis_Simulator(render=False)
        env.reset(seed=seed + rank)
        return env
    return _init

def setup_wandb(config, random_name):
    """Weights & Biases 설정"""
    run = wandb.init(
        project="genesis-robot",
        name=random_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    return run


Is_Genesis_initialized = False
env = Genesis_Simulator(render=False)
env = Monitor(env)

def train_genesis(
    algorithm="SAC",
    total_timesteps=30_000_000,
    seed=42,
    num_envs=1,  # Genesis는 1개 환경만 사용
    learning_rate=0.00056234,
    batch_size=1024,
    n_epochs=10,
    gamma=0.97,
    buffer_size=30_000_000,
    learning_starts=2048,
    train_freq=10,
    gradient_steps=8,
    save_freq=1000,
    eval_freq=10000,
    n_eval_episodes=10,
    log_interval=1,
    device="auto",
    random_name=None
):
    """Genesis Simulator 훈련 함수"""
    # 랜덤 이름 생성
    if random_name is None:
        random_name = get_random_name()

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
    global Is_Genesis_initialized, env

    # wandb 초기화
    run = setup_wandb(config, random_name)
    
    # 실험 결과 저장 디렉토리 생성
    save_path = os.path.join("training_results", random_name)
    os.makedirs(save_path, exist_ok=True)


    # 로컬 로거 설정
    log_path = os.path.join(save_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])


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
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            device=device,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 콜백 설정
    callbacks = [
        WandbCallback(
            model_save_path=None,
            verbose=2,
        ),
        SaveModelEveryNEpisodesCallback(
            n_episodes=save_freq,
            save_path=os.path.join(save_path, "models"),
            prefix_name=random_name,
            wandb_run=run,
            verbose=1
        )
    ]

    # 로거 설정
    model.set_logger(new_logger)

    # 학습 시작
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
        )
    
        # 최종 모델 저장
        final_model_path = os.path.join(save_path, f"{random_name}_final.tar")
        model.save(final_model_path)
        # env.save(os.path.join(save_path, "vec_normalize.pkl"))
        
        # 최종 모델을 W&B에 업로드
        artifact = wandb.Artifact(name=f"{random_name}_final", type="model")
        artifact.add_file(final_model_path)
        run.log_artifact(artifact)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        run.finish()
        
    return model, env

if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # SAC 알고리즘으로 학습 실행
    config = {
        "algorithm": "SAC",
        "total_timesteps": 30_000_000,
        "seed": 42,
        "num_envs": 1,
        "learning_rate": 0.00056234,
        "batch_size": 1024,
        "gamma": 0.97,
        "buffer_size": 30_000_000,
        "learning_starts": 2048,
        "train_freq": 10,
        "gradient_steps": 8,
        "device": device,
    }
    
    # 모델 학습
    model, env = train_genesis(**config)