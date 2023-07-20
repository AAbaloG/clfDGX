import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR


torch.autograd.set_detect_anomaly(True)


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        # self.model = models.resnet50(pretrained=True)
        self.model = models.resnet50(weights=weights)

        # Congelamos los parámetros de la red
        for param in self.model.parameters():
            param.requires_grad = False

        # Modificar la ultima capa FC para adaptarla al datset

        num_classes: int = 5
        num_fts = self.model.fc.in_features
        print("num_fts: ", num_fts)

        self.model.fc = torch.nn.Sequential(torch.nn.Linear(num_fts, 256),
                                            torch.nn.Linear(256, num_classes))

        self.lr = 1e-3

        #
        vector_pesos = torch.load("tensor_pesos_perdida_5.pt")
        self.__criterion = torch.nn.CrossEntropyLoss(weight=vector_pesos)
        #

        # self.__criterion = torch.nn.CrossEntropyLoss()
        self.__step_size = int(78 * 1000 / 2)
        self.__gamma = 0.33

        self.optimizer = None

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.opt = optimizer
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, self.__step_size, gamma=self.__gamma)
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
            }
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['x'].float(), train_batch['y'].float()

        y_hat = self.model(x)

        loss = self.__criterion(y_hat, y.argmax(-1))

        self.log('train_loss', loss)

        current_lr = self.opt.param_groups[0]['lr']
        self.log('learning rate', current_lr)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['x'].float(), val_batch['y'].float()

        y_hat = self.model(x)
        y_hat = y_hat.float()

        loss = self.__criterion(y_hat, y.argmax(-1))

        self.log('val_loss', loss)



