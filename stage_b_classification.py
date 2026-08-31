from torch.utils.data import DataLoader, random_split

def load_fashion_mnist(train_full, test_set):
    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size

    train_set, val_set = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader