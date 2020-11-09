from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
