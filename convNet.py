import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))  # input layer
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.conv3 = nn.Conv2d(64, 128, (5, 5))
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)  # output layer

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = f.max_pool2d(f.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)

        return f.softmax(x, dim=1)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA device found. Using: {torch.cuda.get_device_name(torch.device('cuda:0'))}")
    else:
        print("No devices available!")
        exit(1)

    device = torch.device("cuda:0")
    net = Net().to(device)

    training_data = np.load("train_dataset/training_data.npy", allow_pickle=True)

    image_set = torch.tensor([i[0] for i in training_data], device=device, dtype=torch.float).view(-1, 1, 50, 50)
    image_set = image_set / 255.0

    labels = torch.tensor([i[1] for i in training_data], device=device, dtype=torch.float)

    VAL_PCT = 0.1
    val_size = int(len(image_set) * VAL_PCT)

    train_x = image_set[:-val_size]
    train_y = labels[:-val_size]

    test_x = image_set[-val_size:]
    test_y = labels[-val_size:]

    b_size = 100
    EPOCHS = 250

    train_x.to(device)
    train_y.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fun = nn.MSELoss()

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), b_size)):
            batch_x = train_x[i:i + b_size].view(-1, 1, 50, 50).to(device)
            batch_y = train_y[i:i + b_size].to(device)

            net.zero_grad()

            out = net(batch_x)
            loss = loss_fun(out, batch_y)
            loss.backward()
            optimizer.step()

    print(f"Training finished with loss {loss * 100}%! Saving the model now...")
    torch.save(net.state_dict(), "Brain.pt")

    ##############################################################################

    total = 0
    correct = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_x))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_x[i].view(-1, 1, 50, 50).to(device))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print(f"Accuracy: {correct / total * 100}%")
