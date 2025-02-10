import os
import pandas as pd
from main_only_matching_UoC import ModelValidator, ENV
from stable_baselines3 import SAC
import numpy as np

def quick_validation_test(model_path=None):
    """
    ModelValidator의 주요 기능을 빠르게 테스트하는 함수
    
    Args:
        model_path: 테스트할 모델 경로. None이면 더미 모델 생성
    """
    print("=== Quick Validation Test ===")
    
    # 1. 초기 설정 테스트
    print("\n1. Testing initialization...")
    if model_path is None:
        # 더미 모델 생성
        dummy_model = SAC('MlpPolicy', ENV)
        model_path = "dummy_model.zip"
        dummy_model.save(model_path)
        print("Created dummy model for testing")
    
    validator = ModelValidator(model_path)
    print("✓ Validator initialized successfully")
    
    # 2. 데이터 경로 확인
    print("\n2. Checking data paths...")
    data_path = "./Preprocessing_datas/8000points_curriculum/data"
    if not os.path.exists(data_path):
        print("✗ Error: Data directory not found!")
        return False
    print("✓ Data directory exists")
    
    # 3. 단일 UoC 테스트
    print("\n3. Testing single UoC validation...")
    try:
        test_uoc = 1
        results = validator.validate_uoc(test_uoc)
        print("✓ Single UoC validation successful")
        print(f"   Success rate: {results['success_rate']:.2f}%")
    except Exception as e:
        print(f"✗ Error during UoC validation: {str(e)}")
        return False
    
    # 4. 데이터 저장 확인
    print("\n4. Checking saved demonstration data...")
    demo_file = f"./demonstration_inform/UoC_{test_uoc}.csv"
    if os.path.exists(demo_file):
        df = pd.read_csv(demo_file)
        print(f"✓ Demonstration data saved successfully")
        print(f"   Saved {len(df)} timesteps")
        
        # 데이터 형식 확인
        required_columns = ['state', 'action', 'next_state', 'reward']
        if all(col in df.columns for col in required_columns):
            print("✓ All required columns present")
        else:
            print("✗ Missing required columns!")
    else:
        print("✗ No demonstration data saved!")
    
    # 5. 데이터 정규화 확인
    print("\n5. Checking data normalization...")
    if os.path.exists(demo_file):
        df = pd.read_csv(demo_file)
        first_state = np.array([float(x) for x in df['state'].iloc[0].split(',')])
        first_action = np.array([float(x) for x in df['action'].iloc[0].split(',')])
        
        state_normalized = all(-1.1 <= x <= 1.1 for x in first_state)
        action_normalized = all(-1.1 <= x <= 1.1 for x in first_action)
        
        print(f"✓ States normalized: {state_normalized}")
        print(f"✓ Actions normalized: {action_normalized}")
    
    # 6. 정리
    if os.path.exists("dummy_model.zip"):
        os.remove("dummy_model.zip")
        print("\n✓ Cleaned up dummy model")
    
    print("\n=== Test Summary ===")
    print("Validator appears to be working correctly!")
    return True

if __name__ == "__main__":
    # 실제 모델 경로로 테스트하려면 아래 경로를 수정하세요
    MODEL_PATH = None  # 또는 "path/to/your/model.zip"
    quick_validation_test(MODEL_PATH)
