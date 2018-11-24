import argparse
import os
import pickle
import time
import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm

from data_iterator import DataIterator, PickleDataIterator
from data_reader import DataReader
from models import create_classifier
from utilities import V
from visualise import plot_train_test_accs, plot_train_test_loss


parser = argparse.ArgumentParser(description='Doctor-Patient Interaction Classifier')
parser.add_argument('--saved_models', type=str, default='./saved_models', help='directory to save/load models')
parser.add_argument('--train_data', type=str, default='./data/task_b_interactions_train.tsv', help='train data tsv file')
parser.add_argument('--test_data', type=str, default='./data/task_b_interactions_test.tsv', help='test data tsv file')
parser.add_argument('--num_training_epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--model_name', type=str, default='classifier', help='name of model to train')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
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
        self.create_model(self.encoder_type, self.classifier_type, self.layers, self.drops, params)
    
    def save_model(self):
        path = args.saved_models + '/{0}.pt'.format(self.name)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self):
        path = args.saved_models + '/{0}.pt'.format(self.name)
        self.model.load_state_dict(torch.load(path))
    
    def save_checkpoint(self, save_checkpoint_dir, epoch):
        path = save_checkpoint_dir + '{0}.pt'.format(epoch)
        torch.save(self.model.state_dict(), path)
    
    def create_model(self, encoder_type, classifier_type, layers, drops, params=None):
        self.model = create_classifier(layers, drops, encoder_type, classifier_type, params)

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

        return output.max(0)[1]
    
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


def run_trials(trial_name, n, train_di, test_di, layers, drops, loss_func, opt_func):
    """
    Runs n trials for a given model and stores state_dict checkpoints and statistics
        for each training runs.
    """
    trial_dir = args.saved_models + '/{0}/'.format(trial_name)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
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


def train_model(train_di, test_di, layers, drops, loss_func, opt_func, params=None):
    classifier = ModelWrapper(args.model_name, train_di, test_di, layers, drops, args.classification_type, args.encoder_type, args.classifier_type, params)
    # opt_func = opt_func(classifier.model.parameters(), lr=0.001, weight_decay=0.05)
    opt_func = opt_func(classifier.model.parameters())
    train_accs, test_accs, train_losses, test_losses = classifier.train(args.num_training_epochs, loss_func, opt_func)
    plot_train_test_loss(train_losses, test_losses)
    plot_train_test_accs(train_accs, test_accs)


if __name__ == "__main__":
    # train_di = DataIterator(DataReader(args.train_data), word_embedding_source=args.word_embedding_source, randomise=True) # 405
    # test_di = DataIterator(DataReader(args.test_data), word_embedding_source=args.word_embedding_source, randomise=False) # 135
    # pickle.dump(train_di.data, Path('./data/train_data_glove_50.pkl').open('wb'))
    # pickle.dump(test_di.data, Path('./data/test_data_glove_50.pkl').open('wb'))
    # print(next(iter(train_di)))
    # assert(False)

    train_di = PickleDataIterator('./data/train_data_fasttext_300.pkl', randomise=True)
    test_di = PickleDataIterator('./data/test_data_fasttext_300.pkl', randomise=False)

    if args.classification_type == "simple":
        c = len(train_di.simple_labels)
    else:
        c = len(train_di.extended_labels)
    
    # loss and optimisation functions
    loss_func = nn.CrossEntropyLoss() # NB: this includes a softmax calculation => output logits from my classifiers.
    # opt_func = torch.optim.SGD
    opt_func = torch.optim.Adam

    # For pooling classifier
    layers = [900, 300, 50, c]
    drops = [0, 0, 0]

    # For lstm classifier
    # layers = [150, 50, c]
    # drops = [0, 0]
    # params = {'embedding_dim': 50, 'hidden_dim': 50}

    # run_trials('pool_classifier_sgd', 10, train_di, test_di, layers, drops, loss_func, opt_func)
    # get_trial_best_models('pool_classifier_sgd')
    train_model(train_di, test_di, layers, drops, loss_func, opt_func, params)
