import os
import torch
from datasets import FilteredDataset
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Net import Net


def main():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    batch_size = 8
    epochs = 3000

    class_sample_count: list = []

    folder_classes = sorted(os.listdir("CincoClases/"))
    for c in folder_classes:
        elements: int = len(os.listdir("CincoClases/" + c))
        class_sample_count.append(elements)

    compose = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                 transforms.RandomRotation(30),
                                 transforms.RandomCrop((int(0.8*300), int(0.8*225))),
                                 ])

    train_ds = FilteredDataset(transform=compose, csv_file="./csv/5clases/images_tr_5.csv")
    # test_ds = FilteredDataset(transform=None, csv_file="./paths_csv/images_test_11.csv")
    val_ds = FilteredDataset(transform=None, csv_file="./csv/5clases/images_val_5.csv")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=1)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = Net()

    logger = TensorBoardLogger("tb_logs", name="resnet")

    trainer = pl.Trainer(accelerator="gpu",
                         logger=logger,
                         max_epochs=epochs,
                         log_every_n_steps=25)

    trainer.fit(model, train_dl, val_dl)

    # tensorboard dev upload --logdir tb_logs/

    # https://tensorboard.dev/experiment/q6OMxHGLQveMu3IpguyGEA/#scalars&_smoothingWeight=0 el etto actual (version_0)
    # https://tensorboard.dev/experiment/gclRY6sAQ4Sk2ACGj8Zw5w/#scalars&_smoothingWeight=0&runSelectionState=eyJyZXNuZXQvdmVyc2lvbl8wIjp0cnVlLCJyZXNuZXQvdmVyc2lvbl8xIjp0cnVlfQ%3D%3D (version_1)
    # https://tensorboard.dev/experiment/MMFwN5tKQQ2wJf2FdrkEGQ/ (version2)


if __name__ == '__main__':
    main()
