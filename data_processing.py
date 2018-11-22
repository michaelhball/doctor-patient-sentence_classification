import csv
import pickle

from pathlib import Path


class InteractionsDataReader():
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


if __name__ == "__main__":
    dr = InteractionsDataReader('./data/task_b_interactions_train.tsv')
    dr.read()
    dr.dump('./data')

    data = pickle.load(Path('./data/data.pkl').open('rb'))
    col_headings = pickle.load(Path('./data/col_headings.pkl').open('rb'))

    longest_sent = 0
    for i, d in enumerate(data):
        num_tokens = len(d[3].split())
        if num_tokens > longest_sent:
            longest_sent = num_tokens
    print(longest_sent)

    vocab = set()
