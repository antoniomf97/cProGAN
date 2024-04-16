from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import medmnist
from medmnist import ChestMNIST
from torch.utils.data import DataLoader


print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")


class DataLoaderChestMNIST:
    def __init__(
        self,
        img_size: int = 28,
        batch_size: int = 128,
        download: bool = True,
    ) -> None:
        self.train = ChestMNIST(split="train", download=download, size=img_size)
        self.test = ChestMNIST(split="test", download=download, size=img_size)
        self.val = ChestMNIST(split="val", download=download, size=img_size)

        self.train_loader = DataLoader(
            dataset=self.train, batch_size=batch_size, shuffle=True
        )
        self.train_at_eval_loader = DataLoader(
            dataset=self.train, batch_size=batch_size, shuffle=False
        )
        self.val_loader = DataLoader(
            dataset=self.val, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test, batch_size=batch_size, shuffle=False
        )

    def show_sample(self, n: int=1, m: int=1):
        imgs = [
            Image.fromarray(np.array(self.train[k][0])).convert('RGB') for k in range(n*m)]
        
        _, axes = plt.subplots(n, m, squeeze=False, figsize=(m,n))
        for i in range(n):
            for j in range(m):
                axes[i, j].axis("off")
                axes[i, j].imshow(imgs[j + i*m])
        plt.subplots_adjust(wspace=.0, hspace=.0)
        plt.show()