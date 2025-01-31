import pandas as pd
import numpy as np
from pathlib import Path
import logging
import datetime
import shutil
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SpaceConfig:
    """Configuration for physical space analysis"""
    origin_point: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    x_sections: int = 2  # Divides space into left/right
    y_sections: int = 2  # Divides space into front/back
    z_sections: int = 2  # Divides space into up/down

class PhysicalSpaceAnalyzer:
    """Analyze and categorize points based on physical location"""
    def __init__(self, data_path: str, config: SpaceConfig):
        self.data_path = Path(data_path)
        self.config = config
        self.data = None
        self.xyz_data = None
        self.original_data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess data"""
        logging.info(f"Loading data from {self.data_path}")
        self.original_data = pd.read_csv(self.data_path)
        self.data = self.original_data.copy()
        self.xyz_data = self.data.iloc[:, [0, 1, 2]].copy()
        
        logging.info(f"Data shape: {self.data.shape}")
        logging.info(f"Reference point: {self.config.origin_point}")

    def categorize_points(self) -> None:
        """Categorize points based on their position relative to origin"""
        x_ref, y_ref, z_ref = self.config.origin_point
        
        # Create spatial categories
        self.xyz_data['x_pos'] = np.where(self.xyz_data['x'] > x_ref, 1, 0)
        self.xyz_data['y_pos'] = np.where(self.xyz_data['y'] > y_ref, 1, 0)
        self.xyz_data['z_pos'] = np.where(self.xyz_data['z'] > z_ref, 1, 0)
        
        # Combine positions into single category (0-7)
        self.xyz_data['Category'] = (self.xyz_data['x_pos'] * 4 + 
                                   self.xyz_data['y_pos'] * 2 + 
                                   self.xyz_data['z_pos'])
        
        # Drop temporary columns
        self.xyz_data.drop(['x_pos', 'y_pos', 'z_pos'], axis=1, inplace=True)
        
        # Log category distributions
        category_counts = self.xyz_data['Category'].value_counts().sort_index()
        logging.info("\nCategory distributions:")
        for cat, count in category_counts.items():
            location = []
            if cat & 4: location.append("right")
            else: location.append("left")
            if cat & 2: location.append("back")
            else: location.append("front")
            if cat & 1: location.append("up")
            else: location.append("down")
            logging.info(f"Category {cat} ({' '.join(location)}): {count} points")

class PhysicalSpaceVisualizer:
    """Visualize physical space categorization"""
    def __init__(self, xyz_data: pd.DataFrame, output_dirs: Dict[str, Path]):
        self.xyz_data = xyz_data
        self.output_dirs = output_dirs
        self.categories = sorted(xyz_data['Category'].unique())
        self.colors = plt.cm.jet(np.linspace(0, 1, len(self.categories)))
        
    def plot_all_categories(self, elev: int = 90, azim: int = 0) -> None:
        """Plot all categories in 3D"""
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        for category, color in zip(self.categories, self.colors):
            category_data = self.xyz_data[self.xyz_data['Category'] == category]
            ax.scatter(
                category_data['x'],
                category_data['y'],
                category_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
        
        # Add reference point
        ax.scatter([0], [0], [0.5], c='red', s=100, marker='*')
        
        self._set_plot_parameters(ax, elev, azim, show_legend=False)
        self._save_plot(f'all_regions_{elev}_{azim}.png')
        plt.close()
        
    def plot_individual_categories(self, elev: int = 90, azim: int = 0) -> None:
        """Plot each category separately"""
        for category, color in zip(self.categories, self.colors):
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            
            category_data = self.xyz_data[self.xyz_data['Category'] == category]
            ax.scatter(
                category_data['x'],
                category_data['y'],
                category_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
            
            # Add reference point
            ax.scatter([0], [0], [0.5], c='red', s=100, marker='*')
            
            self._set_plot_parameters(ax, elev, azim, show_legend=False)
            self._save_plot(f'region_{category+1}_{elev}_{azim}.png')
            plt.close()
            
    def plot_sequence(self, elev: int = 90, azim: int = 0) -> None:
        """Plot categories in a sequence"""
        fig = plt.figure(figsize=(60, 15))
        
        for i, (category, color) in enumerate(zip(self.categories, self.colors), 1):
            ax = fig.add_subplot(1, len(self.categories), i, projection='3d')
            
            category_data = self.xyz_data[self.xyz_data['Category'] == category]
            ax.scatter(
                category_data['x'],
                category_data['y'],
                category_data['z'],
                c=[color],
                alpha=0.4,
                s=5
            )
            
            # Add reference point
            ax.scatter([0], [0], [0.5], c='red', s=100, marker='*')
            
            ax.set_title(f'Region {category+1}', fontsize=14)
            self._set_plot_parameters(ax, elev, azim, show_legend=False)
        
        plt.tight_layout()
        self._save_plot(f'sequence_{elev}_{azim}.png')
        plt.close()

    def _set_plot_parameters(self, ax, elev: int, azim: int, show_legend: bool = True) -> None:
        """Set common plot parameters"""
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)
        ax.set_xlim(-1.71, 1.71)
        ax.set_ylim(-1.71, 1.71)
        ax.set_zlim(-1.0, 2.05)
        ax.view_init(elev=elev, azim=azim)
        
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        
        ax.grid(True)
        
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
            bbox_inches='tight',
            pad_inches=0,
            transparent=True
        )
        logging.info(f"Saved plot: {filename}")

class PhysicalSpaceExporter:
    """Export categorized data to CSV files"""
    def __init__(self, xyz_data: pd.DataFrame, original_data: pd.DataFrame, output_dirs: Dict[str, Path]):
        self.xyz_data = xyz_data
        self.original_data = original_data
        self.categories = sorted(xyz_data['Category'].unique())
        self.output_dirs = output_dirs
        
    def export_categories(self) -> None:
        """Export each category to a separate CSV file"""
        self.original_data['Category'] = self.xyz_data['Category']
        
        for category in self.categories:
            category_data = self.original_data[self.original_data['Category'] == category].copy()
            export_data = category_data.iloc[:, [0, 1, 2]]
            
            filename = f'Region_{category+1}.csv'
            export_data.to_csv(
                self.output_dirs['data'] / filename,
                index=False,
                header=False
            )
            
            logging.info(f"Region_{category+1}: {len(category_data)} pieces of data")
            logging.info(f"Saved to: {self.output_dirs['data'] / filename}")

def setup_directories(base_path: Path) -> Dict[str, Path]:
    """Create directory structure for outputs"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_path / f"physical_space_{timestamp}"
    
    dirs = {
        'root': run_dir,
        'logs': run_dir / 'logs',
        'figures': run_dir / 'figures',
        'data': run_dir / 'data',
        'source': run_dir / 'source',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration"""
    log_file = log_dir / "physical_space_analysis.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

def main():
    # Create directory structure
    base_path = Path.cwd() / "Physical_Space_Data"
    output_dirs = setup_directories(base_path)
    
    # Setup logging
    setup_logging(output_dirs['logs'])
    
    # Copy original data file
    try:
        original_data_path = Path('data_points/data_points_origin.csv')
        destination_path = output_dirs['source'] / 'data_points_origin.csv'
        shutil.copy2(original_data_path, destination_path)
        logging.info(f"Successfully copied original data file to: {destination_path}")
    except Exception as e:
        logging.error(f"Error copying original data file: {str(e)}")
        raise

    # Configuration
    config = SpaceConfig(
        origin_point=(0.0, 0.0, 0.5),
        x_sections=2,
        y_sections=2,
        z_sections=2
    )
    
    try:
        # Physical space analysis
        analyzer = PhysicalSpaceAnalyzer(
            'data_points/normalized_data_points.csv',
            config
        )
        analyzer.categorize_points()

        # Export results
        logging.info("\nExporting results")
        exporter = PhysicalSpaceExporter(
            analyzer.xyz_data,
            analyzer.original_data,
            output_dirs
        )
        exporter.export_categories()

        # Visualize results
        logging.info("\nGenerating visualizations")
        visualizer = PhysicalSpaceVisualizer(analyzer.xyz_data, output_dirs)
        visualizer.plot_all_categories(elev=90, azim=0)
        visualizer.plot_all_categories(elev=0, azim=90)
        visualizer.plot_individual_categories(elev=90, azim=0)
        visualizer.plot_individual_categories(elev=0, azim=90)
        visualizer.plot_sequence(elev=90, azim=0)
        visualizer.plot_sequence(elev=0, azim=90)
        
        logging.info("\nAnalysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()