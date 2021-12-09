import torch
import numpy as np
import convNet

class Network:
    def __init__(self, net, device):
        self.device = device
        self.net = net
        self.batch_size = 100
        self.batch_x = []
        self.tensor_x = []

    def load_model(self):
        self.net.load_state_dict(torch.load("Brain.pt"))
        self.net.eval()
        print(self.net)

    def make_batch(self, frame):
        self.batch_x.append(np.array(frame))

        if self.batch_x == self.batch_size:
            self.tensor_x = torch.tensor([i for i in self.batch_x], device=self.device, dtype=torch.float).view(-1, 1, 50, 50).to(self.device)

    def pass_forward(self, frame):
        self.make_batch(frame)

        if self.batch_size >= 100:
            with torch.no_grad():
                for i in range(0, len(self.tensor_x)):
                    net_out = self.net(self.tensor_x[i])
                    predicted_class = torch.argmax(net_out)

                    print(predicted_class)

        self.batch_x.clear()
        self.tensor_x.clear()