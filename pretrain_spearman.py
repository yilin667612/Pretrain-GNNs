import argparse
  
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
import random
from torch_geometric.data import Batch
from audtorch.metrics.functional import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
import pandas as pd
import scipy.stats
from util import MaskAtom
import pickle
from tqdm import tqdm
import numpy as np
from statistics import mean 
from spearman import spearman

from model import GNN 
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        return torch.sum(x*summary, dim = 1)

class Infomax(nn.Module):
    def __init__(self, gnn, discriminator):
        super(Infomax, self).__init__()
        self.gnn = gnn
        self.discriminator = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_max_pool


def train(args, model, device, loader, optimizer,dataset):
    model.train()

    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model.gnn(batch.x, batch.edge_index, batch.edge_attr)
        graph_emb = model.pool(node_emb, batch.batch)

        index = torch.tensor(np.random.choice(range(0,10),len(batch.id)))
        sample_sim = batch.smi[torch.tensor(range(len(batch.id))),index]
        sim_data = dataset[sample_sim[:,0].long()]
        sim_batch = Batch.from_data_list(sim_data).to(device)

        node_emb_sim = model.gnn(sim_batch.x, sim_batch.edge_index, sim_batch.edge_attr)
        graph_emb_sim = model.pool(node_emb_sim, sim_batch.batch)

        cos_sim = F.cosine_similarity(graph_emb, graph_emb_sim, dim=1)
        # print(" cos_sim ", cos_sim)
        tani_sim = sample_sim[:,1]
        # print("tani_sim",tani_sim)
        p = spearman(cos_sim.unsqueeze(0),tani_sim.unsqueeze(0))

        optimizer.zero_grad()
        loss = 1-p
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum/step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--sample_num', type=int, default=4,
                        help='number of sample (default: 1).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'Tanimoto/zinc_data_sim_1000', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    # dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate =0.15, mask_edge=0))

    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    discriminator = Discriminator(args.emb_dim)

    model = Infomax(gnn, discriminator)
    
    model.to(device)
    if not args.input_model_file == "":
        model.gnn.load_state_dict(torch.load(args.input_model_file ))
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # print(optimizer)
    loss_list = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model, device, loader, optimizer,dataset)
        
        print(train_loss)
        loss_list.append(train_loss)

        torch.save(gnn.state_dict(), args.output_model_file + "_ckpt_epoch_{epoch}.pth".format(epoch=epoch))
        # print(gnn.state_dict().keys())

    with open(args.output_model_file, 'wb') as f:
        pickle.dump({"train_loss": np.array(loss_list)},f)

if __name__ == "__main__":
    main()
