from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler
import pickle
from utils import get_lr, AvgMeter

scaler = GradScaler()

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    # tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    model.train()
    for batch in tqdm(train_loader):

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = model(batch)
            loss = torch.mean(loss)

        scaler.scale(loss).backward()

        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = len(batch)
        loss_meter.update(loss.item(), count)

        # tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        scaler.update()

    return loss_meter

def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()

    # tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(batch)
            loss = torch.mean(loss)
            count = len(batch)
            loss_meter.update(loss.item(), count)

        # tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter