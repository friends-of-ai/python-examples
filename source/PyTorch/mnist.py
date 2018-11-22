import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

kwargs = {} # if gpu is disabled
# kwargs = {'num_workers': 1, 'pin_memory: True} # if gpu is enabled
train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(), # first, convert image to PyTorch tensor
                       transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                   ])
    ),
    batch_size=64,
    shuffle=True,
    **kwargs
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(), # first, convert image to PyTorch tensor
                       transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                   ])
    ),
    batch_size=64,
    shuffle=True,
    **kwargs
)

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1,  10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()

        # 320: 20 pictures with 4x4 pixels
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        #print(x.size())
        # batch_size; each picture splitted to 20 pictures with 4x4 pixels
        # torch.Size([64, 20, 4, 4])

        # convert 20 pictures with 4x4 pixels to a linear vector with 320 neurons (20 x 4 x 4)
        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # softmax: only on of each 10 outputs (labels) should be 1; the rest should be 0
        return F.log_softmax(x, dim=1)


model = Netz()

# it's always a good idea to start with lr=0.1 and momentum=0.8
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()

    counter = 0

    # batch_id from 0 to 63, (64 x data, 64 x target -> 0 to 9)
    for batch_id, (data, target) in enumerate(train_data):

        # if we do have a gpu
        # data = data.cuda()
        # target = target.cuda()

        # convert data and target from tensors to variable
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)

        criterion = F.nll_loss

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if counter % (2 * 64) == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_id * len(data),
                len(train_data.dataset),
                100. * batch_id / len(train_data),
                loss.item()
            ))

        counter += 1


def test(epoch):
    model.eval()
    loss = 0
    correct = 0

    for data, target in test_data:

        # if we do have a gpu
        # data = data.cuda()
        # target = target.cuda()

        # convert data and target from tensors to variable
        data = Variable(data)
        target = Variable(target)

        out = model(data)

        loss += F.nll_loss(out, target, reduction='sum').item()

        # 64 x 10 labels (the label with max number -> class)
        #print(out.data)

        prediction = out.data.max(1, keepdim=True)[1]

        correct += prediction.eq(target.data.view_as(prediction)).sum().item()

    loss = loss / len(test_data.dataset)

    print('Test epoch: {}  \tLoss: {:.6f}\tCorrect: {:.2f} %'.format(epoch, loss, 100. * correct / len(test_data.dataset)))
    #print('Correct: ', correct)
    #print('Records all: ', len(test_data.dataset))
    #print('Percent Correct: {:.2f} %'.format(100. * correct / len(test_data.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test(epoch)
