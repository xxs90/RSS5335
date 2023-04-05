from torchmetrics.classification.accuracy import BinaryAccuracy
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy

# these helper functions will all go in a utils file that gets imported
def get_flowers_dataloaders(batch_size, train_tfm=None):
  default_tfm = torchvision.transforms.Compose([
      torchvision.transforms.Resize(128),
      torchvision.transforms.RandomCrop(128),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  if train_tfm is None:
    train_tfm = default_tfm
  else:
    train_tfm = torchvision.transforms.Compose([
        train_tfm,
        default_tfm,
    ])

  train_set = torchvision.datasets.Flowers102('./', 
                                              split='train',
                                              transform=train_tfm,
                                              download=True)
  val_set = torchvision.datasets.Flowers102('./', split='val', 
                                            transform=default_tfm,
                                            download=True)
  train_dl = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                         num_workers=2, persistent_workers=True)
  val_dl = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False,
                                       num_workers=2, persistent_workers=True)
  return train_dl, val_dl

def get_voc_dataloaders(batch_size):
  img_tfm = torchvision.transforms.Compose([
      torchvision.transforms.Resize((128, 128)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  target_tfm = torchvision.transforms.Compose([
      torchvision.transforms.Resize((128, 128)),
      torchvision.transforms.PILToTensor(),
  ])

  train_set = torchvision.datasets.VOCSegmentation('./', year='2008', 
                                                   image_set='train', 
                                                   transform=img_tfm,
                                                   target_transform=target_tfm,
                                                   download=True)
  val_set = torchvision.datasets.VOCSegmentation('./', year='2008',
                                                 image_set='val', 
                                                 transform=img_tfm,
                                                 target_transform=target_tfm,
                                                 download=True)
  train_dl = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                         num_workers=2, persistent_workers=True)
  val_dl = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False,
                                       num_workers=2, persistent_workers=True)
  return train_dl, val_dl

class SegmentationModule(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.acc_metric = BinaryAccuracy()
    self.criterion = nn.BCELoss()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
    
  def training_step(self, batch, batch_idx):
    x, y = batch
    y[y==255] = 0
    y[y>0] = 1
    y_pred = self(x)
    loss = self.criterion(y_pred, y.float())
    self.log("train_loss", loss)
    acc = self.acc_metric(y_pred, y)
    self.log("train_acc", acc, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y[y==255] = 0
    y[y>0] = 1
    y_pred = self(x)
    loss = self.criterion(y_pred, y.float())
    self.log("val_loss", loss)
    acc = self.acc_metric(y_pred, y)
    self.log("val_acc", acc, prog_bar=True)
    return loss

def check_q1a(model):
  errors = 0
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  if num_params != 62378344:
    print('[ERROR]: Your implementation has incorrect number of trainable parameters')
    errors += 1

  img = torch.randn((32, 3, 227, 227)).float()
  out = model(img)
  if out.shape != (32, 1000):
    print('[ERROR]: Your implementation produces incorrect output shape')
    errors +=1 
  if not torch.allclose(out.sum(-1), torch.ones(img.size(0))):
    print('[ERROR]: Your implementation does not produce valid logits')
  
  if errors == 0:
    print('PASSED')


def check_q1b(model):
  errors = 0
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  if num_params != 62378344:
    print('[ERROR]: Your implementation has incorrect number of trainable parameters')
    errors += 1

  img = torch.randn((32, 3, 37, 37)).float()
  out = model(img)
  try:
    out = model(img)
    if out.shape != (32, 1000):
      print('[ERROR]: Your implementation produces incorrect output shape')
      errors +=1 
  except RuntimeError as e:
    print(e)
    print('[ERROR]: Your implementation has shape mismatches in forward pass')
    errors +=1 

  if errors == 0:
    print('PASSED')
