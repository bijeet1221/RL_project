# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:04:53 2025

@author: Bijeet Basak, Lokesh Kumar, Yash Sengupta
"""

import os
import numpy as np
import pandas as pd


class CSVProcessor:
    """
    Reads CSV files from a folder sequentially, splits each file's data into 10 groups,
    computes the average (mean of y-values) and slope (linear fit slope) for each group,
    and then normalizes each of those ten values to the [0,1] range.

    Attributes:
        folder (str): Path to the directory containing CSV files.
        files (List[str]): Sorted list of CSV filenames.
        index (int): Current file index to process.
    """

    def __init__(self, folder: str):
        self.folder = folder
        # List only .csv files and sort by numeric filename
        self.files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith('.csv')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
        )
        self.index = 0

    def _normalize(self, values):
        """
        Linearly scale a list of numbers to the [0,1] range.
        If all values are equal or there's only one non-nan value, returns zeros.
        """
        arr = np.array(values, dtype=float)
        # ignore NaNs when computing min/max
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if np.isclose(vmax, vmin):
            return [0.0] * len(arr)
        return ((arr - vmin) / (vmax - vmin)).tolist()

    def process_next(self):
        """
        Processes the next CSV in the folder.

        Returns:
            norm_avgs (List[float]): Normalized means of y-values for each of the 10 groups.
            norm_slopes (List[float]): Normalized slopes of the best-fit line for each group.

        Raises:
            StopIteration: If there are no more CSV files to read.
        """
        if self.index >= len(self.files):
            self.index = 0
            # raise StopIteration("No more CSV files to process.")

        file_name = self.files[self.index]
        file_path = os.path.join(self.folder, file_name)

        # Read CSV assuming columns 'x' and 'y'
        df = pd.read_csv(file_path)
        x = df['Frequency'].to_numpy()
        y = df['Gain'].to_numpy()
        x_new = []
        for item in x:
            item = item.split('+')
            # print(item)
            item = float(item[0][1:])
            x_new.append(item)
        x_new = np.array(x_new)
        # print(x_new)
        # print(y)

        # Split into 10 nearly equal groups
        x_groups = np.array_split(x_new, 10)
        y_groups = np.array_split(y, 10)

        # Compute raw averages and slopes
        avg_list = [np.nanmean(group_y) for group_y in y_groups]
        slope_list = []
        for xi, yi in zip(x_groups, y_groups):
            if len(xi) < 2 or np.all(np.isnan(yi)):
                slope_list.append(np.nan)
            else:
                m, _ = np.polyfit(xi, yi, deg=1)
                slope_list.append(m)

        # Normalize each list to [0,1]
        norm_avgs = self._normalize(avg_list)
        norm_slopes = self._normalize(slope_list)

        self.index += 1
        return norm_avgs, norm_slopes, y


if __name__ == '__main__':
    folder_path = 'train_graphs/csvs'  # Change to your CSV folder path
    processor = CSVProcessor(folder_path)

    try:
        while True:
            norm_avgs, norm_slopes, target = processor.process_next()
            current_file = processor.files[processor.index - 1]
            print(f"File: {current_file}")
            print("Normalized group averages:", norm_avgs)
            print("Normalized group slopes:", norm_slopes)
            print('-' * 40)
    except StopIteration:
        print("All files processed.")
