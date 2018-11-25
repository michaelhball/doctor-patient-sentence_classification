import csv
import pickle

from pathlib import Path


class DataReader():
    def __init__(self, data_file):
        self.data_file = data_file
    
    def read(self):
        with open(self.data_file) as f:
            reader = csv.reader(f, delimiter='\t')
            self.col_headings = next(reader)
            self.data = [row for row in reader]
            
            return self.data
    
    def dump(self, dir_):
        pickle.dump(self.col_headings, Path(dir_+'/col_headings.pkl').open('wb'))
        pickle.dump(self.data, Path(dir_+'/data.pkl').open('wb'))