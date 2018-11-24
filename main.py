import argparse
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm

from data_iterator import DataIterator, PickleDataIterator
from data_reader import DataReader
from data_writer import DataWriter
from models import create_classifier
from utilities import V
from visualise import plot_train_test_accs, plot_train_test_loss


parser = argparse.ArgumentParser(description='Doctor-Patient Interaction Classifier')
parser.add_argument('--saved_models', type=str, default='./saved_models', help='directory to save/load models')
parser.add_argument('--train_data', type=str, default='./data/task_b_interactions_train.tsv', help='train data tsv file')
parser.add_argument('--test_data', type=str, default='./data/task_b_interactions_test.tsv', help='test data tsv file')
parser.add_argument('--vocab_file', type=str, default='./data/vocab.pkl', help='vocab file')
parser.add_argument('--output_csv', type=str, default='./data/classifications.csv', help='file path to output csv file')
parser.add_argument('--num_training_epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--model_name', type=str, default='classifier', help='name of model to train')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='choice of pretrained embeddings')
parser.add_argument('--classification_type', type=str, default='simple', help='choice of classification task: simple/extended')
parser.add_argument('--encoder_type', type=str, default='max_pool_embeddings', help='choice of sentence encoder')
parser.add_argument('--classifier_type', type=str, default='basic', help='choice of classifier model')
args = parser.parse_args()


class ModelWrapper():
    def __init__(self, name, train_di, test_di, layers, drops, classification_type, encoder_type, classifier_type, params=None):
        self.name = name
        self.train_di, self.test_di = train_di, test_di
        self.simple_labels = self.train_di.simple_labels
        self.extended_labels = self.train_di.extended_labels
        self.classification_type = classification_type
        self.encoder_type = encoder_type
        self.classifier_type = classifier_type
        self.layers, self.drops = layers, drops
        self.params = params
        self.reinitialise()
    
    def save_model(self):
        path = args.saved_models + '/{0}.pt'.format(self.name)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self):
        path = args.saved_models + '/{0}.pt'.format(self.name)
        self.model.load_state_dict(torch.load(path))
    
    def save_checkpoint(self, save_checkpoint_dir, epoch):
        path = save_checkpoint_dir + '{0}.pt'.format(epoch)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, model_file):
        self.model.load_state_dict(torch.load(model_file))

    def reinitialise(self):
        self.model = create_classifier(self.layers, self.drops, self.encoder_type, self.classifier_type, self.params)

    def test_loss_and_accuracy(self, loss_func, load=False):
        if load:
            self.load_model()
        self.model.eval()
        self.model.training = False
        total_loss, total_correct = 0.0, 0
        for i, x in enumerate(iter(self.test_di)):
            class_ = x[4] if self.classification_type == "simple" else x[5]
            output = self.model(V((x[2],x[3])))
            total_loss += loss_func(output, V(class_)).item()
            if output[0].max(0)[1] == class_:
                total_correct += 1
        
        accuracy = total_correct / len(self.test_di)
        avg_loss = total_loss / len(self.test_di)

        return avg_loss, accuracy

    def classify(self, x, load=False):
        if load:
            self.load_model()
        self.model.eval()
        self.model.training = False
        output = self.model(V((x[2],x[3])))

        return output.detach().numpy()[0]
    
    def train(self, num_epochs, loss_func, opt_func, save_checkpoints_dir=None):
        print("---------------  Training Classifier ---------------")
        start_time = time.time()

        self.model.train()
        self.model.training = True
        train_accs, test_accs, train_losses, test_losses = [], [], [], []
        for e in tqdm(range(num_epochs)):
            total_loss, num_correct = 0.0, 0

            for i, x in enumerate(iter(self.train_di)):
                self.model.zero_grad()
                output = self.model(V((x[2], x[3])))

                class_ = x[4] if self.classification_type == "simple" else x[5]
                if output[0].max(0)[1] == class_:
                    num_correct += 1

                loss = loss_func(output, V(class_))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()
            
            test_loss, test_acc = self.test_loss_and_accuracy(loss_func)
            train_losses.append(total_loss/len(self.train_di))
            train_accs.append(num_correct/len(self.train_di))
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if save_checkpoints_dir:
                self.save_checkpoint(save_checkpoints_dir, e+1)

        elapsed_time = time.time() - start_time
        print("Classifier training completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        
        return train_accs, test_accs, train_losses, test_losses


def run_trials(trial_name, n, train_di, test_di, layers, drops, loss_func, opt_func): # UPDATE THIS TO JUST TAKE THE CLASSIFIER IN.
    """
    Runs n trials for a given model and stores state_dict checkpoints and statistics
        for each training runs.
    """
    trial_dir = args.saved_models + '/{0}/'.format(trial_name)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
        metadata = "Parameters: \n classification_type={0} classifier_type={1}\n encoder_type={2}\n opt_func={3}\n  layers={4}\n drops={5}".format(
            classifier.classification_type, classifier.classifier_type, classifier.encoder_type, opt_func, classifier.layers, classifier.drops
        )
        with Path(trial_dir+'metadata.txt').open('wb') as f:
            f.write(metadata)
        for t in range(n):
            run_dir = trial_dir + 'run_{0}/'.format(t+1)
            os.makedirs(run_dir)
            train_di.reset()
            test_di.reset()
            classifier = ModelWrapper(args.model_name, train_di, test_di, layers, drops, args.classification_type, args.encoder_type, args.classifier_type)
            opt_func = opt_func(classifier.model.parameters(), lr=0.01, weight_decay=0.05)
            train_accs, test_accs, train_losses, test_losses = classifier.train(args.num_training_epochs, loss_func, opt_func, run_dir)
            plot_train_test_loss(train_losses, test_losses, display=False, save_file=run_dir+'loss.png')
            plot_train_test_accs(train_accs, test_accs, display=False, save_file=run_dir+'accuracy.png')
            pickle.dump(train_accs, Path(run_dir+'train_accuracies.pkl').open('wb'))
            pickle.dump(test_accs, Path(run_dir+'test_accuracies.pkl').open('wb'))
            pickle.dump(train_losses, Path(run_dir+'train_losses.pkl').open('wb'))
            pickle.dump(test_losses, Path(run_dir+'test_losses.pkl').open('wb'))
    else:
        print('this trial name has already been used')


def get_trial_best_models(trial_name):
    """
    Collects the best model from each run in a given trial and stores
        them in 'best_models' subdirectory of that trial.
    """
    trial_dir = args.saved_models + '/{0}/'.format(trial_name)
    run_dirs = [r[0] for r in os.walk(trial_dir)][1:]
    best_models_dir = trial_dir+'best_models/'
    if not os.path.exists(best_models_dir):
        os.makedirs(best_models_dir)
        for j, run_dir in enumerate(run_dirs):
            test_accs = pickle.load(Path(run_dir+'/test_accuracies.pkl').open('rb'))
            best_acc, idx = 0.0, -1
            for i, a in enumerate(test_accs):
                if a > best_acc:
                    best_acc = a
                    idx = i
            best_model = torch.load(run_dir+'/{0}.pt'.format(idx+1))
            torch.save(best_model, best_models_dir+'{0}.pt'.format(j+1))
    else:
        print('you have already collected the best models for this trial')


def test_trial_ensemble(trial_name, classifier):
    """
    Tests accuracy of ensembled predictions from the saved best models of
        a given trial.
    """
    models_dir = args.saved_models + '/{0}/best_models/'.format(trial_name)
    best_models = [m[2] for m in os.walk(models_dir)][0]
    classifiers = []
    for m in best_models:
        new_classifier = classifier
        new_classifier.load_checkpoint(models_dir+m)
        classifiers.append(new_classifier)
    
    total_correct = 0
    for i, x in enumerate(classifier.test_di):
        label = x[4] if classifier.classification_type == "simple" else x[5]
        predictions = [c.classify(x) for c in classifiers]
        avg_prediction = np.mean(predictions, 0)
        class_prediction = avg_prediction.argmax(0)
        if class_prediction == label:
            total_correct += 1
    
    return total_correct / len(classifier.test_di)


def train_model(classifier, num_epochs, loss_func, opt_func, visualise=False, save=False):
    """
    Trains a model, returns final test accuracy, and optionally visualises 
        training process
    """
    train_accs, test_accs, train_losses, test_losses = classifier.train(num_epochs, loss_func, opt_func)
    if visualise:
        plot_train_test_loss(train_losses, test_losses)
        plot_train_test_accs(train_accs, test_accs)
    if save:
        classifier.save_model()

    return test_accs[-1]


def model_accuracy(n, classifier, num_epochs, loss_func, opt_func, lr):
    """
    Calculates a models classification accuracy on the test dataset as
        an average over the final model for each of n trials.
    """
    final_accuracies = []
    for i in range(n):
        classifier.reinitialise()
        opt = opt_func(classifier.model.parameters(), lr=lr)
        acc = train_model(classifier, num_epochs, loss_func, opt)
        final_accuracies.append(acc)
    
    return round(np.mean(final_accuracies), 3)


def classify_to_csv(encoder_type, classifier_type, simple_classifier, extended_classifier, test_tsv, output_file):
    """
    Takes classifiers for each grain of classification and creates a csv identical
        to the test tsv but with predicted classifications and accuracies.
    """
    if encoder_type == "bow" or encoder_type == "lstm":
        simple = "/{0}_simple.pt".format(encoder_type)
        extended = "/{0}_extended.pt".format(encoder_type)
    elif classifier_type == "pooling":
        if args.word_embedding.startswith("fasttext"):
            simple = "/{0}_{1}_simple.pt".format(encoder_type, "fasttext")
            extended = "/{0}_{1}_extended.pt".format(encoder_type, "fasttext")
        else:
            simple = "/{0}_{1}_simple.pt".format(encoder_type, "glove")
            extended = "/{0}_{1}_extended.pt".format(encoder_type, "glove")
    simple_classifier.load_checkpoint(args.saved_models + simple)
    extended_classifier.load_checkpoint(args.saved_models + extended)
    dw = DataWriter(simple_classifier, extended_classifier, test_tsv, output_file)
    dw.write()


if __name__ == "__main__":
    loss_func = nn.CrossEntropyLoss()
    simple_labels = pickle.load(Path('./data/simple_labels.pkl').open('rb'))
    extended_labels = pickle.load(Path('./data/extended_labels.pkl').open('rb'))
    labels = (simple_labels, extended_labels)
    c = len(simple_labels) if args.classification_type == "simple" else len(extended_labels)
    vocab = pickle.load(Path(args.vocab_file).open('rb'))

    
    ###############################################
    # Construct desired models and parameters
    ###############################################


    if args.encoder_type.endswith("pool_embeddings") and args.classifier_type == "basic":
        # Basic WE-Pool Model
        train_di = PickleDataIterator('./data/train_data_{0}.pkl'.format(args.word_embedding), randomise=True)
        test_di = PickleDataIterator('./data/test_data_{0}.pkl'.format(args.word_embedding))
        layers = [50, 300, c]
        drops = [0, 0]
        params = None
        opt_func = torch.optim.SGD
        lr = 0.01
    elif args.encoder_type == "basic" and args.classifier_type == "pooling":
        # WE-Pool Model
        train_di = PickleDataIterator('./data/train_data_{0}.pkl'.format(args.word_embedding), labels, randomise=True)
        test_di = PickleDataIterator('./data/test_data_{0}.pkl'.format(args.word_embedding), labels)
        if args.classification_type == "simple":
            layers = [900, 200, c] if args.word_embedding == "fasttext_300" else [150, 100 , c]
            drops = [0.2, 0.2] if args.word_embedding == "fasttext_300" else [0, 0]
        else:
            layers = [900, 100, c] if args.word_embedding == "fasttext_300" else [150, 500, c]
            drops = [0, 0]
        params = None
        opt_func = torch.optim.SGD
        lr = 0.01
    elif args.encoder_type == "lstm" and args.classifier_type == "rnn_pooling":
        # LSTM Model
        train_di = PickleDataIterator('./data/train_data_{0}.pkl'.format(args.word_embedding), labels, randomise=True)
        test_di = PickleDataIterator('./data/test_data_{0}.pkl'.format(args.word_embedding), labels)
        layers = [150, 300, c]
        drops = [0, 0]
        params = {'embedding_dim': 50, 'hidden_dim': 100}
        opt_func = torch.optim.Adam
        lr = 0.001
    elif args.encoder_type == "bow" and args.classifier_type == "basic":
        # BoW Model
        train_di = DataIterator(DataReader(args.train_data), randomise=True, bow=True, vocab=vocab, labels=labels) # 405
        test_di = DataIterator(DataReader(args.test_data), randomise=False, bow=True, vocab=vocab, labels=labels) # 135
        if args.classification_type == "simple":
            layers = [769, 300, c]
            drops = [0.7, 0.7]
        else:
            layers = [769, 50, c]
            drops = [0, 0]
        params = {'vocab': vocab}
        opt_func = torch.optim.SGD
        lr = 0.01
    
    else:
        print('not a valid encoder-classifier combination')


    ###############################################
    # PERFORM REQUESTED TASK
    ###############################################


    model_name = args.model_name + "_" + args.classification_type
    classifier = ModelWrapper(model_name, train_di, test_di, layers, drops,
            args.classification_type, args.encoder_type, args.classifier_type, params)
    if args.task == "train":
        opt = opt_func(classifier.model.parameters(), lr)
        print(train_model(classifier, args.num_training_epochs, loss_func, opt, visualise=True, save=False))
    elif args.task == "model_accuracy":
        print(model_accuracy(5, classifier, args.num_training_epochs, loss_func, opt_func, lr))
    elif args.task == "output_csv":
        simple_classifier = classifier
        extended_classifier = ModelWrapper(args.model_name+'_extended', train_di, test_di, [769,50,12], [0,0],
                "extended", args.encoder_type, args.classifier_type, params)
        classify_to_csv(args.encoder_type, args.classifier_type, simple_classifier, extended_classifier, args.test_data, args.output_csv)
        print('successfully created CSV')