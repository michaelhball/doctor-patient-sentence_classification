import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator


def plot_train_test_loss(train_losses, test_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, 'b-', label='train loss')
    ax.plot(test_losses, 'r-', label='test loss')
    ax.legend()
    ax.set(xlabel='epoch number', ylabel='average loss', title='train/test losses')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    plt.show()


def plot_train_test_accs(train_accs, test_accs):
    fig, ax = plt.subplots()
    ax.plot(train_accs, 'b-', label='train accuracy')
    ax.plot(test_accs, 'r-', label='test accuracy')
    ax.set(xlabel='epoch number', ylabel='accuracy', title='train/test accuracies')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    plt.show()
