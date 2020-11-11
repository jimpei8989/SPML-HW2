from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()

        self.X = X
        self.Y = Y.tolist()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


class JointDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        super().__init__()
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __getitem__(self, index):
        return (
            self.dataset_a[index] + (0,)
            if index < len(self.dataset_a)
            else self.dataset_b[index - len(self.dataset_a)] + (1,)
        )

    def __len__(self):
        return len(self.dataset_a) + len(self.dataset_b)
