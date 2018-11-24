import csv

from pathlib import Path

from data_reader import DataReader


class DataWriter():
    def __init__(self, simple_classifier, extended_classifier, test_tsv, output_file):
        """
        Class to create complete output csv, entering classification predictions
        Args:
            simple_classifier (ModelWrapper): Trained classifier on simple classificaton task
            extended_classifier (ModelWrapper): Trained classifier on extended classificaton task
            test_tsv (file): TSV file containing test data to be classified
            output_file (file): Output CSV file path
        """
        self.simple_c, self.extended_c = simple_classifier, extended_classifier
        self.dr = DataReader(test_tsv)
        self.di = self.simple_c.test_di
        self.column_headings = self.dr.col_headings
        self.tsv_data = self.dr.read()
        self.formatted_data = self.di.data
        self.output_file = output_file
        self.output_data = None

    def create_output_data(self):
        self.output_data = [self.column_headings]
        for (x1, x2) in zip(self.tsv_data, self.formatted_data):
            assert(x1[0] == x2[0])
            simple_output = self.simple_c.classify(x2)
            simple_class = simple_output.argmax(0)
            simple_class_str = self.di.simple_labels[simple_class]
            simple_confidence = simple_output[simple_class]
            extended_output = self.extended_c.classify(x2)
            extended_class = extended_output.argmax(0)
            extended_class_str = self.di.extended_labels[extended_class]
            extended_confidence = extended_output[extended_class]
            confidences = "simple: {0}, extended: {1}".format(round(simple_confidence, 3), round(extended_confidence, 3))
            self.output_data.append(x1[:6] + [simple_class_str, extended_class_str, confidences])

    def write(self):
        if not self.output_data:
            self.create_output_data()
        with Path(self.output_file).open('wb') as f:
            writer = csv.writer(f)
            writer.writerows(self.output_data)
