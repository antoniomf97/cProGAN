from dataloader import DataLoaderChestMNIST


def run():
    dataLoader = DataLoaderChestMNIST(img_size=28, batch_size=128)
    dataLoader.show_sample(n=5, m=5)
    
    print(dataLoader.train_loader.dataset)


if __name__ == "__main__":
    run()
