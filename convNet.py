import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.optim.lr_scheduler as sh
from tqdm import tqdm


# Network module for creating ConvNet Inherits from nn.Module (that's where the layers are defined)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))  # input layer
        self.conv2 = nn.Conv2d(32, 64, (5, 5))  # hidden Convolution 1
        self.conv3 = nn.Conv2d(64, 128, (5, 5))  # hidden Convolution 2
        self.fc1 = nn.Linear(128 * 2 * 2, 512)  # Transfer from Convolution to linear 128 * 2 * 2 is the image shape
        self.fc2 = nn.Linear(512, 256)  # hidden linear 1
        self.fc3 = nn.Linear(256, 128)  # hidden linear 2
        self.fc4 = nn.Linear(128, 64)  # hidden linear 3
        self.fc5 = nn.Linear(64, 10)  # output layer
        # 1 input layer 1 out layer 6 hidden layers

    def forward(self, x):
        """Function defines win which order the data pass through the network"""
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)  # pool a 2 x 2 window for data and
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)  # -||-
        x = f.max_pool2d(f.relu(self.conv3(x)), 2)  # -||-
        x = torch.flatten(x, 1)  # flatten 2d conv layers to pass into linear layers
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)

        return f.softmax(x, dim=1)


if __name__ == "__main__":
    # run on gpu if available
    if torch.cuda.is_available():
        print(f"CUDA device found. Using: {torch.cuda.get_device_name(torch.device('cuda:0'))}")
    else:
        print("No devices available!")
        exit(1)

    # get cuda device and send network to gpu
    device = torch.device("cuda:0")
    net = Net().to(device)

    # load training data using numpy into a numpy arr
    training_data = np.load("train_dataset/training_data.npy", allow_pickle=True)

    # convert numpy arrays into pytorch tensors, allocate the tensors to gpu
    image_set = torch.tensor([i[0] for i in training_data], device=device, dtype=torch.float).view(-1, 1, 50, 50)
    image_set = image_set / 255.0

    # load labels
    labels = torch.tensor([i[1] for i in training_data], device=device, dtype=torch.float)

    # test just some percentage of the dataset
    VAL_PCT = 0.1
    val_size = int(len(image_set) * VAL_PCT)

    # training set
    train_x = image_set[:-val_size]
    train_y = labels[:-val_size]

    # test set
    test_x = image_set[-val_size:]
    test_y = labels[-val_size:]

    # make batch sizes and epoch numbers
    b_size = 100
    EPOCHS = 250

    # send train images and labels tensors to gpu
    train_x.to(device)
    train_y.to(device)

    # define and optimizer for neural network
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = sh.ExponentialLR(optimizer, gamma=0.9)

    loss_fun = nn.MSELoss()  # loss function

    #  this is where the main part of the training happens
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), b_size)):
            batch_x = train_x[i:i + b_size].view(-1, 1, 50, 50).to(device)  # make batches for images
            batch_y = train_y[i:i + b_size].to(device)  # make batches for labels

            net.zero_grad()  # zero all gradients

            out = net(batch_x)
            loss = loss_fun(out, batch_y)
            loss.backward()  # back-propagate loss function
            optimizer.step()  # moves the optimizer forward
        scheduler.step()

    print(f"Training finished with loss {loss * 100}%! Saving the model now...")
    torch.save(net.state_dict(), "Brain.pt")

    # TESTING PHASE #
    ##############################################################################

    total = 0
    correct = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_x))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_x[i].view(-1, 1, 50, 50).to(device))[0]
            predicted_class = torch.argmax(net_out)  # return the largest value in the tensor

            if predicted_class == real_class:
                correct += 1
            total += 1

    print(f"Accuracy: {correct / total * 100}%")
