import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dynamixel_utils import Dynamposition2Cablelength_MX64T

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
        # Constants for Dynamixel
        self.offset = 4096
        self.maxlmt = np.array([self.offset + 1517, self.offset + 639, self.offset + 1550])
        self.minlmt = self.maxlmt - 2900
        self.initp = self.maxlmt.copy()
        self.upperlmt = self.maxlmt.copy()
        self.lowerlmt = self.minlmt + 700

    def load_and_preprocess(self):
        """Load .mat file and convert raw data to initial DataFrame."""
        print(f"Loading data from {self.file_path}...")
        data = scipy.io.loadmat(self.file_path)

        # Extract relevant data
        dxl_goal_position = data['dxl_goal_position']
        motiondata = data['Motiondata'][:, 0:3]  # Select columns 0 to 2 (X, Y, Z)

        # Calculate cable lengths
        N = dxl_goal_position.shape[1]
        cable_lengths = np.zeros((3, N), dtype=float)

        for i in range(N):
            Dynap_i = dxl_goal_position[:, i]
            cblen_i = Dynamposition2Cablelength_MX64T(Dynap_i, self.initp)
            cable_lengths[:, i] = cblen_i

        # Normalize motion data
        origin = motiondata[0, :]
        normalized_motiondata = motiondata - origin

        # Create DataFrame
        self.df = pd.DataFrame({
            'cblen1': cable_lengths[0, :],
            'cblen2': cable_lengths[1, :],
            'cblen3': cable_lengths[2, :],
            'X': normalized_motiondata[:, 0],
            'Y': normalized_motiondata[:, 1],
            'Z': normalized_motiondata[:, 2]
        })
        print("Data loaded and initial preprocessing done.")

    def _detect_and_replace_outliers(self, data, threshold=20):
        """Helper method to detect and replace outliers."""
        filtered_data = np.copy(data)
        outliers_indices = []
        outliers_values = []

        for idx in range(1, len(data)):
            if np.abs(filtered_data[idx] - filtered_data[idx - 1]) > threshold:
                outliers_indices.append(idx)
                outliers_values.append(filtered_data[idx])
                filtered_data[idx] = filtered_data[idx - 1]

        return filtered_data, np.array(outliers_indices), np.array(outliers_values)

    def clean_data(self, alpha=0.7):
        """Apply outlier removal and low-pass filtering."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        for coord in ['X', 'Y', 'Z']:
            # Outlier detection
            self.df[coord], outliers_idx, outliers_values = self._detect_and_replace_outliers(self.df[coord].values)
            if len(outliers_values) > 0:
                print(f"Outliers in {coord}: {len(outliers_values)} detected.")
            
            # Low-pass filter
            filtered_signal = np.zeros(self.df[coord].shape)
            filtered_signal[0] = self.df[coord][0]
            for i in range(1, len(self.df[coord])):
                filtered_signal[i] = alpha * self.df[coord][i] + (1 - alpha) * filtered_signal[i - 1]
            self.df[coord] = filtered_signal
        
        print("Data cleaning (outlier removal + low-pass filter) completed.")

    def save_to_excel(self, output_path):
        """Save the processed DataFrame to an Excel file."""
        if self.df is None:
            raise ValueError("No data to save.")
        
        self.df.to_excel(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def plot_results(self):
        """Plot the X, Y, Z coordinates."""
        if self.df is None:
            raise ValueError("No data to plot.")

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']

        for i, ax in enumerate(axs):
            coord = labels[i]
            ax.plot(self.df[coord], label=coord, color=colors[i])
            ax.set_title(f'{coord} Coordinate')
            ax.set_ylabel('Value')
            ax.legend()

        axs[2].set_xlabel('Sample Index')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Configuration
    INPUT_MAT_FILE = './Datacollection/ExplorationResult20250416.mat'
    OUTPUT_EXCEL_FILE = 'Processed_Data3w_20250416.xlsx'

    # Run pipeline
    processor = DataProcessor(INPUT_MAT_FILE)
    processor.load_and_preprocess()
    processor.clean_data(alpha=0.7)
    processor.save_to_excel(OUTPUT_EXCEL_FILE)
    processor.plot_results()
