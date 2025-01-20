import gc
from itertools import product
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import datetime
import time
import shutil
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging
import warnings
import json
from pathlib import Path

def setup_directories(base_path: Path) -> Dict[str, Path]:
    """Create directory structure for outputs and copy original data file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_path / f"{timestamp}"
    
    # Create subdirectories
    dirs = {
        'root': run_dir,
        'logs': run_dir / 'logs',
        'figures': run_dir / 'figures',
        'data': run_dir / 'data',
        'source': run_dir / 'source',
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration"""
    log_file = log_dir / "clustering_analysis.log"
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Python 3.8+ 에서 기존 설정을 강제로 덮어씁니다
    )

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters"""
    n_clusters: int = 8
    feature_columns: List[int] = None
    feature_weights: List[float] = None
    selected_algorithm: Optional[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [3, 7, 8]
        if self.feature_weights is None:
            self.feature_weights = [1.5, 1.0, 1.0]

class ClusteringAnalyzer:
    """Main class for performing clustering analysis"""
    def __init__(self, data_path: str, config: ClusteringConfig, available_algorithms: List[str]):
        self.data_path = Path(data_path)
        self.config = config
        self.available_algorithms = available_algorithms
        self.data = None
        self.feature_data = None
        self.xyz_data = None
        self.clusters = {}
        self.cluster_centers = {}
        self.labels = {}
        self.scores = {}
        self.best_algorithm = None
        self.original_data = None
        
        self._load_data()
        self._setup_algorithms()
        
    def _load_data(self) -> None:
        """Load and preprocess data"""
        logging.info(f"Loading data from {self.data_path}")
        self.original_data = pd.read_csv(self.data_path)
        self.data = self.original_data.copy()
        
        logging.info(f"Data shape: {self.data.shape}")
        logging.info(f"Selected feature columns: {self.config.feature_columns}")
        logging.info(f"Feature weights: {self.config.feature_weights}")
        
        self.xyz_data = self.data.iloc[:, [0, 1, 2]].copy()
        self.feature_data = self.data.iloc[:, self.config.feature_columns].copy()
        
        for col, weight in zip(self.feature_data.columns, self.config.feature_weights):
            self.feature_data[col] *= weight
            logging.info(f"Applied weight {weight} to column {col}")

    def find_optimal_gaussian_parameters(self, X: np.ndarray) -> Dict[str, Dict]:
        """Find optimal parameters for Gaussian mixture models using grid search"""
        json_path = Path('optimized_gaussian_parameters.json')
        
        # JSON 파일이 존재하면 저장된 파라미터 사용
        if json_path.exists():
            logging.info("Loading existing optimal parameters from JSON file...")
            with open(json_path, 'r') as f:
                return json.load(f)
        
        logging.info("\nStarting parameter optimization for Gaussian mixture models...")
        
        # 파라미터 그리드 정의
        param_grid = {
            'covariance_type': ['full', 'tied', 'diag'], 
            'max_iter': [300, 500, 1000],
            'n_init': [10, 20],
            'tol': [1e-4, 1e-5],
            'reg_covar': [1e-6, 1e-7],
            'init_params': ['k-means++', 'random'],
            'warm_start': [False],
            'verbose': [0]
        }
        
        # BayesianGaussianMixture 전용 파라미터
        bayesian_params = {
            'weight_concentration_prior_type': ['dirichlet_process'],
            'weight_concentration_prior': [0.01, 1.0],
            'mean_precision_prior': [0.1, 1.0],
            'degrees_of_freedom_prior': [None, self.config.n_clusters + 2]
        }
        
        best_params = {
            'GaussianMixture': {},
            'BayesianGaussianMixture': {}
        }
        best_scores = {
            'GaussianMixture': float('-inf'),
            'BayesianGaussianMixture': float('-inf')
        }
        
        # 총 조합 수 계산
        base_combinations = len([p for p in product(*param_grid.values())])
        bayesian_combinations = len([p for p in product(*bayesian_params.values())])
        
        logging.info(f"Total combinations - GMM: {base_combinations}, BGMM: {base_combinations * bayesian_combinations}")

        max_tries = 1000  # Early stopping limit

        for model_name in best_params.keys():
            logging.info(f"\nOptimizing parameters for {model_name}...")
            
            param_combinations = list(product(*param_grid.values()))
            if model_name == 'BayesianGaussianMixture':
                param_combinations = list(product(
                    param_combinations,
                    *bayesian_params.values()
                ))
            
            total_tried = 0
            
            for params in param_combinations:
                if total_tried >= max_tries:
                    logging.info(f"Early stopping after {max_tries} trials")
                    break
                    
                # 현재 파라미터 조합 생성
                if model_name == 'BayesianGaussianMixture':
                    base_params, *bayesian_specific = params
                else:
                    base_params = params
                    
                current_params = {
                    'n_components': self.config.n_clusters,
                    'covariance_type': base_params[0],
                    'max_iter': base_params[1],
                    'n_init': base_params[2],
                    'tol': base_params[3],
                    'reg_covar': base_params[4],
                    'init_params': base_params[5],
                    'warm_start': base_params[6],
                    'verbose': base_params[7],
                    'random_state': 42
                }
                
                # Bayesian 전용 파라미터 추가
                if model_name == 'BayesianGaussianMixture':
                    current_params.update({
                        'weight_concentration_prior_type': bayesian_specific[0],
                        'weight_concentration_prior': bayesian_specific[1],
                        'mean_precision_prior': bayesian_specific[2],
                        'degrees_of_freedom_prior': bayesian_specific[3]
                    })
                
                try:
                    # 모델 초기화 및 학습
                    model = GaussianMixture(**current_params) if model_name == 'GaussianMixture' else BayesianGaussianMixture(**current_params)
                    
                    start_time = time.time()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        model.fit(X)
                        
                        # 시간 초과 체크 (60초)
                        if time.time() - start_time > 60:
                            logging.warning(f"Skipping due to timeout: {current_params}")
                            continue
                        
                        # 경고가 없었는지 확인
                        if not any("convergence" in str(warning.message) for warning in w):
                            # 모델 평가
                            labels = model.predict(X)
                            if len(np.unique(labels)) > 1:
                                silhouette = silhouette_score(X, labels)
                                bic_score = -model.bic(X) if hasattr(model, 'bic') else 0
                                combined_score = 0.7 * silhouette + 0.3 * (bic_score / abs(bic_score) if bic_score != 0 else 0)
                                
                                if combined_score > best_scores[model_name]:
                                    best_scores[model_name] = combined_score
                                    best_params[model_name] = current_params
                                    
                                    # 로깅
                                    logging.info(f"New best parameters found for {model_name}:")
                                    logging.info(f"Parameters: {current_params}")
                                    logging.info(f"Combined score: {combined_score:.4f}")
                                    logging.info(f"Silhouette score: {silhouette:.4f}")
                                    if bic_score != 0:
                                        logging.info(f"BIC score: {-bic_score:.4f}")
                                    
                                    # 중간 결과 저장
                                    with open(json_path, 'w') as f:
                                        json.dump(best_params, f, indent=4)
                
                except Exception as e:
                    logging.warning(f"Error with parameters {current_params}: {str(e)}")
                    continue
                
                # 진행상황 추적 및 메모리 관리
                total_tried += 1
                if total_tried % 50 == 0:
                    logging.info(f"Tried {total_tried} combinations for {model_name}")
                    gc.collect()
    
        # 최종 결과 저장
        logging.info("\nSaving final optimal parameters to JSON file...")
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        return best_params
    
    def _setup_algorithms(self) -> None:
        """Initialize clustering algorithms with optimized parameters"""
        # 먼저 기본 KMeans 관련 알고리즘 설정
        all_algorithms = {
            'KMeans': KMeans(
                n_clusters=self.config.n_clusters,
                n_init=10,
                random_state=42
            ),
            'BisectingKMeans': BisectingKMeans(
                n_clusters=self.config.n_clusters,
                random_state=42
            ),
            'MiniBatchKMeans': MiniBatchKMeans(
                n_clusters=self.config.n_clusters,
                random_state=42
            )
        }
        
        # Gaussian 모델들의 최적 파라미터 찾기
        X = self.feature_data.to_numpy()
        optimal_params = self.find_optimal_gaussian_parameters(X)
        
        # 최적화된 파라미터로 Gaussian 모델들 추가
        all_algorithms['GaussianMixture'] = GaussianMixture(**optimal_params['GaussianMixture'])
        all_algorithms['BayesianGaussianMixture'] = BayesianGaussianMixture(**optimal_params['BayesianGaussianMixture'])
        
        # 선택된 알고리즘만 사용
        self.algorithms = {name: all_algorithms[name] for name in self.available_algorithms}
        logging.info(f"Using selected algorithms: {', '.join(self.algorithms.keys())}")
        
        # 각 알고리즘의 주요 파라미터 로깅
        for name, algo in self.algorithms.items():
            logging.info(f"\n{name} parameters:")
            for param, value in algo.get_params().items():
                logging.info(f"  {param}: {value}")

    def _calculate_dunn_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Dunn Index for cluster validation with proper handling of empty clusters"""
        distances = pdist(X)
        dist_matrix = squareform(distances)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0
        
        min_inter_cluster_distance = np.inf
        max_intra_cluster_distance = 0
        
        for i in unique_labels:
            # Get points in current cluster
            cluster_i_points = X[labels == i]
            
            if len(cluster_i_points) > 1:
                # Calculate intra-cluster distance only if cluster has more than 1 point
                intra_dist = np.max(pdist(cluster_i_points))
                max_intra_cluster_distance = max(max_intra_cluster_distance, intra_dist)
            
            for j in unique_labels[unique_labels > i]:
                # Get points in other cluster
                cluster_j_points = X[labels == j]
                
                if len(cluster_i_points) > 0 and len(cluster_j_points) > 0:
                    # Calculate inter-cluster distance only if both clusters have points
                    inter_dist = np.min(dist_matrix[np.ix_(labels == i, labels == j)])
                    min_inter_cluster_distance = min(min_inter_cluster_distance, inter_dist)
        
        if max_intra_cluster_distance == 0 or min_inter_cluster_distance == np.inf:
            return 0.0
            
        return min_inter_cluster_distance / max_intra_cluster_distance

    def evaluate_clusters(self) -> Dict[str, Tuple[float, ...]]:
        logging.info(f"\nEvaluate clustering results using multiple metrics")
        X = self.feature_data.to_numpy()
        scores = {}
        
        for name, labels in self.labels.items():
            if len(set(labels)) > 1:
                start_time = datetime.datetime.now()
                logging.info(f"Running evaluation with {name}")
                
                # scores[name] = (1.0, 1.0, 1.0, 1.0)
                scores[name] = (
                    silhouette_score(self.feature_data, labels),
                    davies_bouldin_score(self.feature_data, labels),
                    calinski_harabasz_score(self.feature_data, labels),
                    self._calculate_dunn_index(X, labels)
                )
                
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                logging.info(f"{name} Done - Elapsed time: {elapsed_time.total_seconds():.2f} seconds")

        return scores

    def _normalize_scores(self) -> Dict[str, List[float]]:
        """Normalize evaluation scores"""
        normalized = {algo: list(scores) for algo, scores in self.scores.items()}
        
        # 각 평가 지표별 정규화
        metric_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Dunn']
        
        for i in range(4):
            values = [scores[i] for scores in self.scores.values()]
            min_val, max_val = min(values), max(values)
            
            if max_val == min_val:
                continue
                
            logging.info(f"\n Normalized {metric_names[i]} scores:")
            for algo in normalized:
                if i == 1:  # Davies-Bouldin score (lower is better)
                    normalized[algo][i] = (max_val - self.scores[algo][i]) / (max_val - min_val)
                else:  # Other scores (higher is better)
                    normalized[algo][i] = (self.scores[algo][i] - min_val) / (max_val - min_val)
                logging.info(f"{algo}: {normalized[algo][i]:.4f}")
        
        return normalized

    def _select_best_algorithm(self) -> None:
        """Select the best performing algorithm"""
        normalized_scores = self._normalize_scores()
        
        avg_scores = {
            algo: np.mean(scores) 
            for algo, scores in normalized_scores.items()
        }
        
        self.best_algorithm = max(avg_scores.items(), key=lambda x: x[1])[0]
        self.xyz_data['Cluster'] = self.labels[self.best_algorithm]

    def run_clustering(self) -> None:
        logging.info(f"\nExecute clustering analysis")
        X = self.feature_data.to_numpy()
        
        for name, algorithm in self.algorithms.items():
            start_time = datetime.datetime.now()
            logging.info(f"Running clustering with {name}")
            
            self.clusters[name] = algorithm.fit(X)
            
            self.labels[name] = (
                algorithm.labels_ if hasattr(algorithm, 'labels_')
                else algorithm.fit_predict(X)
            )
            
            self.cluster_centers[name] = (
                algorithm.cluster_centers_ if hasattr(algorithm, 'cluster_centers_')
                else algorithm.predict_proba(X)
            )
            
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            logging.info(f"Done - Elapsed time: {elapsed_time.total_seconds():.2f} seconds")
            
        self.scores = self.evaluate_clusters()
        
        # Log raw scores
        logging.info("\nRaw clustering evaluation scores:")
        for algo, scores in self.scores.items():
            logging.info(f"\n{algo}:")
            logging.info(f"  Silhouette score: {scores[0]:.4f}")
            logging.info(f"  Davies-Bouldin score: {scores[1]:.4f}")
            logging.info(f"  Calinski-Harabasz score: {scores[2]:.4f}")
            logging.info(f"  Dunn index: {scores[3]:.4f}")
        
        # Normalize and log normalized scores
        self._select_best_algorithm()
        
        logging.info(f"\nBest performing algorithm: {self.best_algorithm}")
        # Log cluster sizes
        cluster_sizes = Counter(self.labels[self.best_algorithm])
        logging.info(f"Cluster sizes for {self.best_algorithm}:")
        for cluster_id, size in sorted(cluster_sizes.items()):
            logging.info(f"  Cluster {cluster_id}: {size} points")

class RelativeLevelDefiner:
    """Define relative levels based on cluster centers"""
    def __init__(self, clustering_analyzer: ClusteringAnalyzer):
        self.xyz_data = clustering_analyzer.xyz_data
        self.cluster_centers = clustering_analyzer.cluster_centers[clustering_analyzer.best_algorithm]
        self.best_algorithm = clustering_analyzer.best_algorithm
        self.distances = None
        
    def calculate_relative_levels(self) -> None:
        """Calculate and assign relative levels to clusters"""
        self.distances = np.linalg.norm(self.cluster_centers, axis=1)
        sorted_indices = np.argsort(self.distances)
        
        level_mapping = {old: new for new, old in enumerate(sorted_indices)}
        self.xyz_data['Cluster'] = self.xyz_data['Cluster'].map(level_mapping)
        
        # Log distances for each UoC
        logging.info(f"\nCluster distances for {self.best_algorithm}:")
        for old_idx, new_idx in level_mapping.items():
            logging.info(f"UoC {new_idx + 1} (original cluster {old_idx}): distance = {self.distances[old_idx]:.4f}")
        
        # Additional statistics about distances
        logging.info("\nDistance statistics:")
        logging.info(f"Mean distance: {np.mean(self.distances):.4f}")
        logging.info(f"Std distance: {np.std(self.distances):.4f}")
        logging.info(f"Min distance: {np.min(self.distances):.4f}")
        logging.info(f"Max distance: {np.max(self.distances):.4f}")

class DataExporter:
    """Export clustered data to CSV files"""
    def __init__(self, xyz_data: pd.DataFrame, original_data: pd.DataFrame, output_dirs: Dict[str, Path]):
        self.xyz_data = xyz_data
        self.original_data = original_data
        self.clusters = sorted(xyz_data['Cluster'].unique())
        self.output_dirs = output_dirs
        
    def _create_output_directory(self) -> Path:
        """Create output directory for results"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = Path.cwd() / "Uoc_data"
        output_path = base_path / f"UoC_{len(self.clusters)}_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
        
    def export_clusters(self) -> None:
        """Export each cluster to a separate CSV file with only x,y,z coordinates"""
        self.original_data['Cluster'] = self.xyz_data['Cluster']
        
        for cluster in self.clusters:
            cluster_data = self.original_data[self.original_data['Cluster'] == cluster].copy()
            export_data = cluster_data.iloc[:, [0, 1, 2]]
            
            filename = f'UoC_{cluster+1}.csv'
            export_data.to_csv(
                self.output_dirs['data'] / filename, 
                index=False, 
                header=False
            )
            
            logging.info(f"UoC_{cluster+1}: {len(cluster_data)} pieces of data")
            logging.info(f"Saved to: {self.output_dirs['data'] / filename}")

class ClusterVisualizer:
    """Visualize clustering results"""
    def __init__(self, xyz_data: pd.DataFrame, output_dirs: Dict[str, Path]):
        self.xyz_data = xyz_data
        self.output_dirs = output_dirs
        self.clusters = sorted(xyz_data['Cluster'].unique())
        self.colors = plt.cm.jet(np.linspace(0, 1, len(self.clusters)))
        
    def plot_all_clusters(self, elev: int = 90, azim: int = 0) -> None:
        """Plot all clusters in 3D with legend"""
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster, color in zip(self.clusters, self.colors):
            cluster_data = self.xyz_data[self.xyz_data['Cluster'] == cluster]
            ax.scatter(
                cluster_data['x'], 
                cluster_data['y'], 
                cluster_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
        
        self._set_plot_parameters(ax, elev, azim, show_legend=False)
        self._save_plot(f'all_clusters_{elev}_{azim}.png')
        plt.close()

    def plot_individual_clusters(self, elev: int = 90, azim: int = 0) -> None:
        """Plot each cluster separately without legend"""
        for cluster, color in zip(self.clusters, self.colors):
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            
            cluster_data = self.xyz_data[self.xyz_data['Cluster'] == cluster]
            ax.scatter(
                cluster_data['x'],
                cluster_data['y'],
                cluster_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
            
            self._set_plot_parameters(ax, elev, azim, show_legend=False)
            self._save_plot(f'cluster_{cluster+1}_{elev}_{azim}.png')
            plt.close()

    def plot_sequence(self, elev: int = 90, azim: int = 0) -> None:
        """Plot clusters in a sequence without legend"""
        fig = plt.figure(figsize=(60, 15))
        
        for i, (cluster, color) in enumerate(zip(self.clusters, self.colors), 1):
            ax = fig.add_subplot(1, len(self.clusters), i, projection='3d')
            
            cluster_data = self.xyz_data[self.xyz_data['Cluster'] == cluster]
            ax.scatter(
                cluster_data['x'],
                cluster_data['y'],
                cluster_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
            
            ax.set_title(f'UoC {cluster+1}', fontsize=14)
            self._set_plot_parameters(ax, elev, azim, show_legend=False)
            
        plt.tight_layout()
        self._save_plot(f'sequence_{elev}_{azim}.png')
        plt.close()
        
    def _set_plot_parameters(self, ax, elev: int, azim: int, show_legend: bool = False) -> None:
        """Set common plot parameters"""
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        ax.set_xlim(-1.71, 1.71)
        ax.set_ylim(-1.71, 1.71)
        ax.set_zlim(-1.0, 2.05)
        ax.view_init(elev=elev, azim=azim)
        
        # 눈금선은 유지하고 숫자 표기만 제거
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        
        # 격자선 표시
        ax.grid(True)
        
        # 3D 축 면 채우기 제거
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if show_legend:
            legend = ax.legend(fontsize=14, bbox_to_anchor=(1.15, 0.5), loc='center left')
            legend.set_bbox_to_anchor((1.2, 0.5))
        
    def _save_plot(self, filename: str) -> None:
        """Save plot to file"""
        plt.savefig(
            self.output_dirs['figures'] / filename,
            dpi=400,
            bbox_inches='tight',  # Reduces extra whitespace
            pad_inches=0,         # Minimizes padding around the figure
            transparent=True      # Makes the background transparent
        )
        logging.info(f"Saved plot: {filename}")

def main():
    # Create directory structure
    base_path = Path.cwd() / "Uoc_data"
    output_dirs = setup_directories(base_path)
    
    # Setup logging first
    setup_logging(output_dirs['logs'])
    
    # Copy original data file with logging
    try:
        original_data_path = Path('data_points/data_points_origin.csv')
        destination_path = output_dirs['source'] / 'data_points_origin.csv'
        shutil.copy2(original_data_path, destination_path)
        logging.info(f"Successfully copied original data file to: {destination_path}")
    except Exception as e:
        logging.error(f"Error copying original data file: {str(e)}")
        raise

    available_algorithms = [
        'KMeans',
        'BisectingKMeans',
        'MiniBatchKMeans',
        'GaussianMixture',
        'BayesianGaussianMixture'
    ]
    
    # Log analysis start
    logging.info("Starting clustering analysis")
    logging.info(f"Output directory: {output_dirs['root']}")
    
    # Configuration
    config = ClusteringConfig(
        n_clusters=8,
        feature_columns=[3, 7, 8],
        feature_weights=[1.5, 1.0, 1.0]
    )
    
    # Log configuration
    logging.info("\nConfiguration:")
    logging.info(f"Number of clusters: {config.n_clusters}")
    logging.info(f"Feature columns: {config.feature_columns}")
    logging.info(f"Feature weights: {config.feature_weights}")
    logging.info(f"Selected algorithms: {', '.join(available_algorithms)}")

    try:
        # Clustering analysis
        analyzer = ClusteringAnalyzer(
            'data_points/normalized_data_points.csv', 
            config,
            available_algorithms
        )
        analyzer.run_clustering()

        # Define relative levels
        logging.info("\nCalculating relative levels")
        level_definer = RelativeLevelDefiner(analyzer)
        level_definer.calculate_relative_levels()

        # Export results
        logging.info("\nExporting results")
        exporter = DataExporter(
            level_definer.xyz_data, 
            analyzer.original_data,
            output_dirs
        )
        exporter.export_clusters()

        # Visualize results
        logging.info("\nGenerating visualizations")
        visualizer = ClusterVisualizer(level_definer.xyz_data, output_dirs)
        visualizer.plot_all_clusters(elev=90, azim=0)
        visualizer.plot_all_clusters(elev=0, azim=90)
        visualizer.plot_individual_clusters(elev=90, azim=0)
        visualizer.plot_individual_clusters(elev=0, azim=90)
        visualizer.plot_sequence(elev=90, azim=0)
        visualizer.plot_sequence(elev=0, azim=90)

        
        
        logging.info("\nAnalysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()