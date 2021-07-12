import torch
import torch.nn as nn
import time
import argparse

# seq : windows 內的label sequence
# Device configuration
device = torch.device("cpu")


# 跟train的generate不一樣 只返回把每一個line變成list而已
def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = []
    # hdfs = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            #print(ln)
            ln = ln + [-1] * (window_size + 1 - len(ln))
            #print(ln)
            hdfs.append(tuple(ln))
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    # Hyperparameters
    num_classes = 23
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=500.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=5, type=int)
    parser.add_argument('-num_candidates', default=5, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate('openstack_normal2.log_preprocess_session_numberID')
    test_abnormal_loader = generate('openstack_abnormal.log_preprocess_session_numberID')
    #test_abnormal_loader = generate('session_abnormal.log')
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            #一個line 就是一個session,只要一個key不符合prediction我們就說他是anomaly
            #將一個session切成好幾個time windows 每一個time windows都可以進行一次預測
            #這邊的positive是偵測到異常
            #這邊需要去一個一個檢查predicted有沒有在range裡面無法平行運算
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                if label[0] == -1:
                    continue;
                output = model(seq)
                print("-----------------normal ----")
                print("seq",seq)
                print("label",label)
                
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                print("normal predicted set",predicted)
                print("FP",FP)
                if label not in predicted:
                    print("FP + 1")
                    FP += 1
                    break
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                if label[0] == -1:
                    continue;
                output = model(seq)
                print("---------------abnormal ---")
                print("seq",seq)
                print("label",label)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                print("abnormal predicted set",predicted)
                print("TP ",TP)
                if label not in predicted:
                    print("!!!!!!!!!!!!!!!!!!!!TP + 1 !!!!!!!!!!!")
                    TP += 1
                    break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    print('true positive {} ,false negative {},false positive {}'.format(TP,FN,FP))
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    print(' Precision: {:.3f}%, Recall: {:.3f}'.format( P, R))
    print('Finished Predicting')
