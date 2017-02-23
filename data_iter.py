import numpy as np
import random

class DataIterator:
    def __init__(self, data, batch_size):
        assert len(self.data) % batch_size == 0
        self.data = data
        self.batch_size = batch_size
        self.iter = self.make_random_iter()
        
    def next_batch(self):
        try:
            idxs = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = next(self.iter)
            
        X, Y = zip(*[self.data[i] for i in idxs])
        X = np.array(X).T
        Y = np.array(Y).T
        return X, Y

    def make_random_iter(self):
        n = len(self.data)
        shuffled_indexes = np.array(range(n))
        random.shuffle(shuffled_indexes)
        batch_indexes = [shuffled_indexes[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        return iter(batch_indexes)

data = [([1, 2], [11, 22]), ([3, 4], [33, 44]), ([5, 6], [55, 66])]
it = DataIterator(data, 3)

for i in range(4):
    print("batch " + str(i))
    X, Y = it.next_batch()
    print(X.T)
    print('')
    print(Y.T)
    print('')
