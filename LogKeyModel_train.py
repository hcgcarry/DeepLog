import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device);

###output: dataset,[[[一個window的log key]],[[這個window t時 的key]]]
## 所有session的window都混合在同一個list
def generate(filePath):
    num_sessions = 0
    inputs = []
    outputs = []
    with open( filePath, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(filePath, num_sessions))
    print('Number of seqs({}): {}'.format(filePath, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        
        #print("x.shape",x.shape)
        #print("x.size(0)",x.size(0))
        #x shape torch.Size([2048,10,1])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        #print("out",out)
        #print("out shape",out.shape))
        #out shape torch.Size([2048, 10, 64])
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    

    # Hyperparameters
    num_classes = 23
    num_epochs = 500
    batch_size = 2048
    input_size = 1
    model_dir = 'model'
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=5, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate("data/openstack_normal1.log_preprocess_session_numberID")
    #seq_dataset = generate("data/query.log_normal_afterPreProcess_session_numberID")
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            #據說clone .detach()等同於torch.tensor
            #seq shape torch.Size([2048, 10])
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            #seq shape torch.Size([2048, 10, 1])
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        ## train_loss/total_step 平均一個window的loss
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
