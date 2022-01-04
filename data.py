import torch
import matplotlib.pyplot as plt

def mnist():
    # exchange with the corrupted mnist dataset
    from numpy import load

    data = load('C:/Users/Durita Kvilt/OneDrive/Menneske orienteret kunstig intelligens/Machine learning operations/dtu_mlops/data/corruptmnist/test.npz')
    lst = data.files

    test = load('C:/Users/Durita Kvilt/OneDrive/Menneske orienteret kunstig intelligens/Machine learning operations/dtu_mlops/data/corruptmnist/test.npz')
    train = load('C:/Users/Durita Kvilt/OneDrive/Menneske orienteret kunstig intelligens/Machine learning operations/dtu_mlops/data/corruptmnist/train_0.npz')


    return train, test

# mnist()