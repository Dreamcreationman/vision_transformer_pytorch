import os
import time
import torch
from torch import nn
import torch.optim as optim
from util import log
from torch.autograd import Variable
from model import VisionTransformer
from dataset import ImagenetDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

root_path = "E:\Dataset\ImageNet 2012 DataSets"
logger_path = "log/"
checkpoints_path = "checkpoints/"
batch_size = 128
image_size = 256
patch_size = 16
num_layers = 8
num_head = 8
mlp_dim = 2048
dim_model = 512
num_class = 1000
channel = 3
dropout = 0.5
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.1
epoches = 5

logger = log.get_logger(logger_path, "train.log")
tensorboard_log = os.path.join(logger_path, "/tensorboard")
if not os.path.exists(tensorboard_log):
    os.mkdir(tensorboard_log)
writer = SummaryWriter(tensorboard_log)


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size, pad_if_needed=True),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    transforms.ToTensor()
])
train_set = ImagenetDataset(root_path, transform, train=True)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 若能使用cuda，则使用cuda
model = VisionTransformer(image_size,
                        patch_size,
                        batch_size,
                        dim_model,
                        num_head,
                        num_layers,
                        num_class,
                        channel,
                        dropout=dropout)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

logger.info("Start Traing……")
for epoch in range(epoches):
    epoch_losses = []
    epoch_acces = []

    for num_iter, data in enumerate(trainloader):
        iter_loss = 0.0
        iter_acc = 0.0
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if num_iter % 2000 == 0:
            num_correct = (outputs.argmax(dim=1) == labels).float().mean()
            iter_acc += num_correct / len(trainloader)
            epoch_acces.append(iter_acc)
            iter_loss += float(loss.item())
            epoch_losses.append(iter_loss)
            logger.info(
                "Epoch : {}/{} - Iter : {}/{} - Iter loss : {:.4f} - Iter acc: {:.4f}\n"
                    .format(epoch + 1, epoches, num_iter + 1, len(trainloader) / batch_size, iter_loss, iter_acc)
            )
            writer.add_scalar('Train/Loss', loss.item(), num_iter + 1)
            writer.add_scalar('Train/acc', iter_acc, num_iter + 1)
            writer.flush()

        if num_iter % 4000 == 0:
            if not os.path.exists(checkpoints_path):
                os.mkdir(checkpoints_path)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'epoch_losses': epoch_losses, 'epoch_acces': epoch_acces}
            checkpoints_name = "checkpoint" + time.strftime("%m-%d_%H:%M", time.localtime()) + ".pth"
            torch.save(state, os.path.join(checkpoints_path, checkpoints_name))
            logger.info("Save checkpoint: {}……".format(checkpoints_name))

    with torch.no_grad():
        logger.info("Save Model……")
        torch.save(model, 'vit.pth')  # 保存模型