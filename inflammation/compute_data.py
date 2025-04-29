"""Module containing mechanism for calculating standard deviation between datasets.
"""

import json
import glob
import os
import numpy as np

from inflammation import models, views

# better idea: define a unique DataSource class with common features (load_inflammation_data and data_dir)
class CSVDataSource:
    """Class accepting the data directory having CSV files and loading the data contained in it."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Returns a list of 2D NumPy arrays with inflammation data.
        
        Data is loaded from all inflammation CSV files found in a specified directory path."""
        # filepath
        data_file_paths = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)

        return list(data)
        # alternative
        # return [models.load_csv(fname) for fname in data_file_paths]

def load_json(filename):
    """Load a numpy array from a JSON document.
    
    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]
class JSONDataSource:
    """Class accepting the data directory having JSON files and loading the data contained in it."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        # filepath
        data_file_paths = glob.glob(os.path.join(self.data_dir, '*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)

        return data

def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    data = data_source.load_inflammation_data()


    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    views.visualize(graph_data)