import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.optim as optim
import time, sys
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

batch_size = 16
num_workers = 2
shuffle = True
epoch = 100
lr = 0.0001
out_model_fn = 'model/model'

tr_x = np.load('processed_data/train_x.npy')
tr_y = np.load('processed_data/train_y.npy')
te_x = np.load('processed_data/test_x.npy')
te_y = np.load('processed_data/test_y.npy')

class Data2Torch(Dataset):
    def __init__(self, data):
        self.x = data[0].astype(np.float)
        self.y = data[1].astype(np.float)

    def __getitem__(self, index):
    	x = torch.from_numpy(self.x[index]).float()
    	y = torch.from_numpy(np.array([self.y[index]])).float()
    	return x, y

    def __len__(self):
        return len(self.x)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(27, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, 1),
		)

	def forward(self, inp):
		out = self.model(inp)
		return out


def loss_func(pred, tar):
	loss_func = nn.BCEWithLogitsLoss()
	loss = loss_func(pred, tar)
	return loss

class Trainer:
    def __init__(self, model, lr, epoch, save_fn):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn

        print('Start Training #Epoch:%d'%(epoch))

    def fit(self, tr_loader, va_loader):
        st = time.time()
        opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        save_dict = {}
        best_loss = 1000000000

        for e in range(1, self.epoch+1):
            loss_total = 0
            acc_score = 0
            self.model.train()
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))

            # training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()  
                inp, target = Variable(_input[0]), Variable(_input[1])
                pred = self.model(inp)
                loss_train = loss_func(pred, target)  
                loss_train.backward()
                opt.step()

            # Validate
            for batch_idx, _input in enumerate(va_loader):
                inp, target = Variable(_input[0]), Variable(_input[1])
                pred = self.model(inp)
                loss_val = loss_func(pred, target)  

                pred_bin = torch.sigmoid(pred)
                pred_bin[pred_bin>0.5]=1
                pred_bin[pred_bin<=0.5]=0
                acc_score += accuracy_score(pred_bin.detach().numpy(), target.detach().numpy())
                    
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] Loss_train %4f  Loss_val %4f  acc_score %4f  Time %d'
                    %(e, self.epoch, batch_idx+1, len(tr_loader), loss_train, loss_val, acc_score/len(va_loader), time.time() - st))
            sys.stdout.flush()
            print ('\n')

            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn)
                best_loss = loss_val


t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,'drop_last': True}
v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch([tr_x, tr_y]), **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch([te_x, te_y]), **v_kwargs)

model = Net()
if torch.cuda.is_available():
    model.cuda()

Trer = Trainer(model, lr, epoch, out_model_fn)
Trer.fit(tr_loader, va_loader)
