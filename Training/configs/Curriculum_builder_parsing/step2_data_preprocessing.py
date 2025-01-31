import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

class PathManager:
    def __init__(self, base_path: str = "data_points"):
        """데이터와 로그 파일의 경로를 관리하는 클래스"""
        self.base_path = base_path
        self.log_path = os.path.join(base_path, 'logs')
        self._create_directories()
        
        # 파일 경로 설정
        self.origin_csv_path = os.path.join(base_path, 'data_points_origin.csv')
        self.outlier_removed_csv_path = os.path.join(base_path, 'outlier_removed_data_points.csv')
        self.lower_outliers_path = os.path.join(self.log_path, 'lower_outliers.csv')
        self.upper_outliers_path = os.path.join(self.log_path, 'upper_outliers.csv')
        self.normalized_csv_path = os.path.join(base_path, 'normalized_data_points.csv')
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for path in [self.base_path, self.log_path]:
            if not os.path.exists(path):
                os.makedirs(path)

class OutlierRemover:
    def __init__(self, filepath: str):
        """아웃라이어를 제거하는 클래스 초기화
        
        Args:
            filepath: CSV 파일 경로
        """
        self.origin_data = pd.read_csv(filepath)
        self.data = None
        self.lower_outliers = None
        self.upper_outliers = None
    
    def remove_outliers(self, column_index: int) -> None:
        """지정된 컬럼의 IQR을 계산하고 아웃라이어가 있는 행을 제거
        
        Args:
            column_index: 아웃라이어를 검사할 컬럼의 인덱스
        """
        col_values = self.origin_data.iloc[:, column_index]
        
        # IQR 계산
        Q1 = col_values.quantile(0.25)
        Q3 = col_values.quantile(0.75)
        IQR = Q3 - Q1
        
        # 경계값 계산
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 아웃라이어 필터링
        self.lower_outliers = self.origin_data[col_values < lower_bound]
        self.upper_outliers = self.origin_data[col_values > upper_bound]
        self.data = self.origin_data[(col_values >= lower_bound) & (col_values <= upper_bound)]
    
    def save_data(self, output_filepath: str) -> None:
        """아웃라이어가 제거된 데이터를 CSV 파일로 저장
        
        Args:
            output_filepath: 저장할 CSV 파일 경로
        """
        self.data.to_csv(output_filepath, index=False)
        print(f"아웃라이어가 제거된 CSV 파일이 생성되었습니다 -> {output_filepath}")

    def save_outliers_data(self, lower_outlier_path: str, upper_outlier_path: str) -> None:
        """아웃라이어로 판별된 데이터를 CSV 파일로 저장
        
        Args:
            lower_outlier_path: 하한 경계 아웃라이어 데이터를 저장할 경로
            upper_outlier_path: 상한 경계 아웃라이어 데이터를 저장할 경로
        """
        self.lower_outliers.to_csv(lower_outlier_path, index=False)
        print(f"하한 경계 아웃라이어 데이터가 저장되었습니다 -> {lower_outlier_path}")
        self.upper_outliers.to_csv(upper_outlier_path, index=False)
        print(f"상한 경계 아웃라이어 데이터가 저장되었습니다 -> {upper_outlier_path}")
    
    def print_statistics(self) -> None:
        """데이터 처리 통계 출력"""
        print(f"원본 데이터 길이: {len(self.origin_data)}")
        print(f"아웃라이어 제거된 데이터 길이: {len(self.data)}")
        print(f"하한 경계 아웃라이어 수: {len(self.lower_outliers)}")
        print(f"상한 경계 아웃라이어 수: {len(self.upper_outliers)}")

class DataNormalizer:
    def __init__(self, filepath: str):
        """데이터 정규화를 수행하는 클래스 초기화
        
        Args:
            filepath: CSV 파일 경로
        """
        self.filepath = filepath
        self.data = self.load_data()
        self.column_names = [
            'x', 'y', 'z', 'distance', 'joint_delta_6ea', 'joint_delta_3ea',
            'joint1_delta', 'joint2_delta', 'joint3_delta', 'joint4_delta',
            'joint5_delta', 'joint6_delta'
        ]

    def load_data(self) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        return pd.read_csv(self.filepath)

    def normalize_columns(self, normalized_filepath: str, num_of_data: Optional[int] = None) -> None:
        """4번째 컬럼부터 마지막 컬럼까지 Min-Max 정규화 수행
        
        Args:
            normalized_filepath: 정규화된 데이터를 저장할 파일 경로
            num_of_data: 저장할 데이터 수 (None인 경우 전체 데이터 저장)
        """
        scaler = MinMaxScaler()
        self.data.iloc[:, 3:] = scaler.fit_transform(self.data.iloc[:, 3:])
        self.save_data(normalized_filepath, num_of_data)
    
    def save_data(self, normalized_filepath: str, num_of_data: Optional[int] = None) -> None:
        """정규화된 데이터를 CSV 파일로 저장
        
        Args:
            normalized_filepath: 저장할 파일 경로
            num_of_data: 저장할 데이터 수
        """
        self.data.columns = self.column_names
        
        if num_of_data is None:
            self.data.to_csv(normalized_filepath, index=False)
        else:
            sampled_data = self.data.sample(n=num_of_data, random_state=42)
            sampled_data.to_csv(normalized_filepath, index=False)
            print(f"데이터 수가 변경되었습니다: {len(self.data)} -> {num_of_data}")
        
        print(f"정규화된 CSV 파일이 생성되었습니다 -> {normalized_filepath}")
        print(self.data)

def main():
    # 경로 관리자 초기화
    path_manager = PathManager()
    
    # 아웃라이어 제거
    remover = OutlierRemover(path_manager.origin_csv_path)
    remover.remove_outliers(column_index=3)  # distance 컬럼 기준
    remover.save_data(path_manager.outlier_removed_csv_path)
    remover.save_outliers_data(path_manager.lower_outliers_path, path_manager.upper_outliers_path)
    remover.print_statistics()
    
    # 데이터 정규화
    normalizer = DataNormalizer(path_manager.outlier_removed_csv_path)
    normalizer.normalize_columns(path_manager.normalized_csv_path)

if __name__ == "__main__":
    main()