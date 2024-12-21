from datasets import load_dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import lightning as L
from datetime import datetime
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import PyTorchModelHubMixin

IMG_SIZE = 224

config = {
        "batch_size": 64,
        "lr": 3e-4, 
        "epochs": 100, 
        "architecture": "CNN", 
        "dataset": "Stanford Cars",
        "huggingface-dataset-repo-id": "tanganke/stanford_cars",
        "accumulate_grad_batches": 4,
        "val_check_interval": 1.0,
        "gradient_clip_val": 1.0,
        "deterministic": True,
        "summary_depth": 3,
        # "num_workers": 3,
        "pin_memory": True,
        "model_store_path": "models/", 
        "model_save_top_k": 3,
        "model_save_monitor_metric": "validation/loss",
        "model_save_huggingface_repo_id": "ball1433/stanford-cars-resnet",
        "load_model_before_training": False, # load the pre-trained model from the model_save_huggingface_repo_id
}

# define a wandb logger
wandb.init(project="resnet-stanford-cars", name=f"experiment-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", config=config)
wandb_logger = WandbLogger()
dataset_artifact = wandb.Artifact(name="stanford-cars-dataset", type="dataset")
model_artifact = wandb.Artifact(name="resnet-stanford-cars-model", type="model")



# install stanford cars dataset
train_dataset = load_dataset(config.get("huggingface-dataset-repo-id", "tanganke/stanford_cars"), split="train")
test_dataset = load_dataset(config.get("huggingface-dataset-repo-id", "tanganke/stanford_cars"), split="test")

# define collate_fn to apply to dataset
# resize the image
# check if image is rgb. if not, convert to rgb
# add some variation to make the model robust
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
def train_collate_fn(samples):
    
    images = []
    for image in [sample["image"] for sample in samples]:
        # convert to rgb if not.
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = train_transform(image)
        images.append(image)

    images = torch.stack(images)

    labels = [torch.tensor(sample["label"]) for sample in samples]
    labels = torch.stack(labels)

    return images, labels

def test_collate_fn(samples):
    images = []
    for image in [sample["image"] for sample in samples]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = test_transform(image)
        images.append(image)

    images = torch.stack(images)

    labels = [torch.tensor(sample["label"]) for sample in samples]
    labels = torch.stack(labels)


    return images, labels


class CNN_Classifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(16*(IMG_SIZE // 8)*(IMG_SIZE // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 196)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Lit_CNN_Classifier(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.config = config
        self.batch_size=config.get("batch_size", 10)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)

        train_loss = self.loss(outputs, labels)

        self.log("train/loss", train_loss, on_epoch=True)

        return train_loss 
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        # calculate the validation loss
        val_loss = self.loss(outputs, labels)
        self.log("validation/loss", val_loss, on_epoch=True)

        _, predictions = torch.max(outputs, dim=1)

        # calculate the accuracy
        correct_cnt = 0
        total_cnt = 0

        columns = ["image", "prediction", "label"]
        data = []
        for image, label, prediction in zip(images, labels, predictions):
            if label == prediction:
                total_cnt += 1
                correct_cnt += 1
            else:
                total_cnt += 1

            data.append([wandb.Image(image), prediction, label])
        wandb_logger.log_table(key="validation/visualization", columns=columns, data=data)

            
        val_acc = correct_cnt / total_cnt
        self.log("validation/accuracy", val_acc, on_epoch=True)

        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.config.get("lr", 3e-4))
        return optimizer

    def train_dataloader(self):
        # return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=self.config.get("num_workers", 1), pin_memory=self.config.get("pin_memory", False))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_collate_fn, pin_memory=self.config.get("pin_memory", False))


    def val_dataloader(self):
        # return DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=test_collate_fn, num_workers=self.config.get("num_workers", 1), pin_memory=self.config.get("pin_memory", False))
        return DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=test_collate_fn, pin_memory=self.config.get("pin_memory", False))



# Define callbacks
class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing the model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(config.get("model_save_huggingface_repo_id"))

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing the model to the hub after training is done")
        pl_module.model.push_to_hub(config.get("model_save_huggingface_repo_id"))

early_stop_callback = EarlyStopping(monitor="validation/accuracy", patience=10, verbose=False, mode="max")

checkpoint_callback = ModelCheckpoint(dirpath="models/", save_top_k=3, monitor="validation/loss", filename='{epoch}-{val_loss:.2f}', save_on_train_epoch_end=True)

# define a Trainer
trainer = L.Trainer(
    max_epochs=config.get("epochs", 100),
    accumulate_grad_batches=config.get("accumulate_grad_batches", 8),
    val_check_interval=config.get("val_check_interval", 1.0),
    gradient_clip_val=config.get("gradient_clip_val", 1.0),
    num_sanity_val_steps=5,
    callbacks=[PushToHubCallback(), early_stop_callback],
    deterministic=config.get("deterministic", True),
    logger=wandb_logger,
)

model = CNN_Classifier()
model_module = Lit_CNN_Classifier(model, config)

# do the training
trainer.fit(model_module)
