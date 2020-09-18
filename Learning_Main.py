import os
import pickle
import time
import glob
from Learning_Model import GAT
from utils import load_data, accuracy,load_data2
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

feats, adj_mat, labels, idx_train, idx_val, labels_train, labels_test = load_data(True)
# features, adj, labels, idx_train, idx_val, labels_train, labels_test = load_data2()
epochs = 1000
fast_mode = True
patience = 10000

model = GAT(nfeat=feats.shape[1],
            nhid=8,
            nclass=int(labels.max()) + 1,
            dropout=0.2,
            nheads=2,
            alpha=0.2)
# optimizer = optim.Adam(model.parameters(),lr=1,weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=50, verbose=True)

features, adj, labels = Variable(torch.tensor(feats)), Variable(adj_mat), Variable(torch.tensor(labels))
model.double()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

cg = {'train': idx_train,
      'validation': idx_val,
      'x': feats,
      'adj': adj_mat
      }
with open('BestModel/train_validation.pickle', 'wb') as handle:
    pickle.dump(cg, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fast_mode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    lr_scheduler.step(loss_train)

    return acc_val


# Train model
t_total = time.time()
acc_values = []
bad_counter = 0
best = 0
best_epoch = 0
best_model_state_dict = None

for epoch in range(epochs):
    acc_values.append(train(epoch))

    if acc_values[-1] > best:
        best_model_state_dict = model.state_dict()
        best = acc_values[-1]
        best_epoch = epoch
        bad_counter = 0
        torch.save(best_model_state_dict, 'BestModel/model.pth.tar')
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



with open('BestModel/train_validation.pickle', 'wb') as handle:
    pickle.dump(cg, handle, protocol=pickle.HIGHEST_PROTOCOL)
torch.save(best_model_state_dict, 'BestModel/model.pth.tar')