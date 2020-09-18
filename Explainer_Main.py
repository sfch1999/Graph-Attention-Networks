import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_data
from Explainer_Model import Explainer
from Learning_Model import GAT
from torch.autograd import Variable

with open('Pickles/preds.pickle', 'rb') as handle:
    labels = pickle.load(handle).astype(np.long)
with open('BestModel/train_validation.pickle', 'rb') as handle:
    cg_dict = pickle.load(handle)

train_idx = cg_dict['train']
test_idx = cg_dict['validation']
X = cg_dict['x']
adj = cg_dict['adj']

X, adj = Variable(torch.tensor(X)), Variable(adj)


node_idx = 1000

labels = torch.tensor(labels, dtype=torch.long)
labels_train = labels[train_idx]
labels_test = labels[test_idx]

model = GAT(nfeat=X.shape[1],
            nhid=8,
            nclass=int(labels.max()) + 1,
            dropout=0.2,
            nheads=2,
            alpha=0.2)
model.double()

model.load_state_dict(torch.load('BestModel/model.pth.tar'))

explainer = Explainer(
    model=model,
    adj=adj,
    feat=X,
    label=labels,
    pred=model(X, adj),
    train_idx=train_idx,
    test_idx=test_idx,
    print_training=True,
    graph_mode=False,
)
mask = explainer.explain_graph()

with open('BestModel/feat_mask.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.imshow(X * torch.tensor(mask))
plt.colorbar()
plt.show()

# TODO: networkx

# TODO: GAT and this one also
