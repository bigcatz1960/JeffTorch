class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)

        for i in range(0, len(idxs), self.batch_size):
            batch = [self.dataset[j] for j in idxs[i:i+self.batch_size]]
            yield batch
