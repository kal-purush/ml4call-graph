"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
from cgi import test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import numpy as np
import pandas as pd

from tqdm import tqdm

"""
    For GCNs
"""

# def train_epoch_sparse(model, optimizer, device, graph, train_edges, batch_size, epoch, test_neg_df ,monet_pseudo=None):

#     model.train()
    
#     train_edges = train_edges.to(device)
#     total_loss = total_examples = 0
#     neg_df = test_neg_df
#     neg_src = neg_df['src'].tolist()
#     neg_dst = neg_df['dst'].tolist()

#     start_index = 0 

#     for perm in tqdm(DataLoader(range(train_edges.size(0)), batch_size, shuffle=True)):
#         # print("perm ===> ", perm)
#         optimizer.zero_grad()
        
#         graph = graph.to(device)
#         x = graph.ndata['feat'].to(device)
#         e = graph.edata['feat'].to(device).float()
        
#         if monet_pseudo is not None:
#             e = monet_pseudo.to(device)
        
#         # Compute node embeddings
#         try:
#             x_pos_enc = graph.ndata['pos_enc'].to(device)
#             sign_flip = torch.rand(x_pos_enc.size(1)).to(device)
#             sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
#             x_pos_enc = x_pos_enc * sign_flip.unsqueeze(0)
#             h = model(graph, x, e, x_pos_enc) 
#         except:
#             h = model(graph, x, e)
        
#         # Positive samples
#         edge = train_edges[perm].t()
#         pos_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
        
#         # Just do some trivial random sampling
#         # edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
#         # neg_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
#         # print(edge[0])

#         end_index = start_index+len(perm)
#         # print("end_index ===> ", end_index)
#         src_edge = neg_src[start_index:end_index]
#         dst_edge = neg_dst[start_index:end_index]

#         neg_out = model.edge_predictor(h[src_edge], h[dst_edge])
#         start_index= start_index+len(perm)
        
#         loss = model.loss(pos_out, neg_out)

#         loss.backward()
#         optimizer.step()

#         num_examples = pos_out.size(0)
#         total_loss += loss.detach().item() * num_examples
#         total_examples += num_examples

#     return total_loss/total_examples, optimizer




def train_epoch_sparse(model, optimizer, device, graph, train_edges, batch_size, epoch, test_neg_df ,monet_pseudo=None):

    model.train()
    
    train_edges = train_edges.to(device)
    
    total_loss = total_examples = 0
    start_index = 0 

    for perm in tqdm(DataLoader(range(train_edges.size(0)), batch_size, shuffle=True)):
        # print("perm ===> ", perm)
        optimizer.zero_grad()
        
        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()
        
        if monet_pseudo is not None:
            # Assign e as pre-computed pesudo edges for MoNet
            e = monet_pseudo.to(device)
        
        # Compute node embeddings
        try:
            x_pos_enc = graph.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(x_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            x_pos_enc = x_pos_enc * sign_flip.unsqueeze(0)
            h = model(graph, x, e, x_pos_enc) 
        except:
            h = model(graph, x, e)
        
        # Positive samples
        edge = train_edges[perm].t()
        pos_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
        
        # Just do some trivial random sampling
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
        neg_out = model.edge_predictor( h[edge[0]], h[edge[1]] )
        # print("size===>",edge.size())
        loss = model.loss(pos_out, neg_out)

        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.detach().item() * num_examples
        total_examples += num_examples

    return total_loss/total_examples, optimizer


def evaluate_network_sparse(model, device, graph, pos_train_edges, 
                     pos_valid_edges, neg_valid_edges, 
                     pos_test_edges, neg_test_edges, 
                     evaluator, batch_size, epoch, monet_pseudo=None, DATASET_NAME="", MODEL_NAME=""):
    
    model.eval()
    with torch.no_grad():

        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()
        
        if monet_pseudo is not None:
            # Assign e as pre-computed pesudo edges for MoNet
            e = monet_pseudo.to(device)

        # Compute node embeddings
        try:
            x_pos_enc = graph.ndata['pos_enc'].to(device)
            h = model(graph, x, e, x_pos_enc) 
        except:
            h = model(graph, x, e)

        pos_train_edges = pos_train_edges.to(device)
        pos_valid_edges = pos_valid_edges.to(device)
        neg_valid_edges = neg_valid_edges.to(device)
        pos_test_edges = pos_test_edges.to(device)
        neg_test_edges = neg_test_edges.to(device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edges.size(0)), batch_size):
            edge = pos_train_edges[perm].t()
            pos_train_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)
        
        # print("pos train pred======>")
        # print(pos_train_pred)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = pos_valid_edges[perm].t()
            pos_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = neg_valid_edges[perm].t()
            neg_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = pos_test_edges[perm].t()
            pos_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        # print("size ", pos_test_edges.size(0))
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edges.size(0)), batch_size):
            edge = neg_test_edges[perm].t()
            # print("edge ==> ",edge, len(edge))
            neg_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    
    # print(pos_train_pred)
    test_label = torch.ones(len(pos_train_pred))
    # print('Accuracy', ((pos_train_pred >= 0.5) == test_label).sum().item() / len(pos_train_pred))
    # print('validation AUC', compute_auc(pos_valid_pred, neg_valid_pred))
    # print('test AUC', compute_auc(pos_test_pred, neg_test_pred))
    # print("test pos pred====>>>", len(pos_test_edges), len(pos_test_pred))

    root_path ="results/"
    path = root_path+'NEW_DATASET--'+DATASET_NAME+"/"+MODEL_NAME

    out_csv_file = open(path+'/POS_PRED/pred_score_with_'+str(epoch)+"_pos.csv","w+")
    for i in range(len(pos_test_edges)):
        out_csv_file.write(str(pos_test_edges[i][0].item())+","+str(pos_test_edges[i][1].item())+","+str(pos_test_pred[i].item())+"\n")
    
    out_csv_file = open(path+'/NEG_PRED/pred_score_'+str(epoch)+"_neg.csv","w+")
    for i in range(len(neg_test_edges)):
        out_csv_file.write(str(neg_test_edges[i][0].item())+","+str(neg_test_edges[i][1].item())+","+str(neg_test_pred[i].item())+"\n")

    train_accuracy = ((pos_train_pred >= 0.5) == test_label).sum().item() / len(pos_train_pred)
    validation_auc = compute_auc(pos_valid_pred, neg_valid_pred)
    test_auc = compute_auc(pos_test_pred, neg_test_pred)

    from sklearn.metrics import roc_curve
    from scipy.special import expit
    scores = torch.cat([pos_test_pred, neg_test_pred]).numpy()
    # scores = expit(scores)

    labels = torch.cat(
        [torch.ones(pos_test_pred.shape[0]), torch.zeros(neg_test_pred.shape[0])]).numpy()
    fpr1, tpr1, thresh1 = roc_curve(labels, scores, pos_label=1)
    
    initial_threshld = thresh1[0]
    # print(tpr1[:-10])
    # print(len(pos_test_edges),"========>>>>>>>>" ,len(pos_test_pred))
    lst = []
    for i in range(len(tpr1)):
        if tpr1[i]>=0.999:
            last_threshld=thresh1[i]
            break
    print("last ======>", last_threshld)


    test_scores = expit(pos_test_pred)
    test_label = torch.ones(len(pos_test_pred))
    # print('Accuracy ====> ', ((test_scores >= 0.55) == test_label).sum().item() / len(test_scores))
    test_pos_accuracy = ((test_scores >= 0.55) == test_label).sum().item() / len(test_scores)

    test_scores = expit(neg_test_pred)
    test_label = torch.zeros(len(neg_test_pred))
    # print('Neg Accuracy ====> ', ((test_scores >= 0.55) == test_label).sum().item() / len(test_scores))
    test_neg_accuracy = ((test_scores >= 0.55) == test_label).sum().item() / len(test_scores)


    # print("scores ====> ", scores)
    with open(path+'/THRESHOLD/threshold_values_with_'+str(epoch)+".txt","w+") as out_file:
        out_file.write(str(thresh1))
        out_file.write("\n")
        out_file.write(str(fpr1))
        out_file.write("\n")
        out_file.write(str(tpr1))
    # print("thresdhold ===> ", thresh1)
    
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='GNN')

    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(path+'/ROC_CURVE/ROC_with_'+str(epoch),dpi=300)
    # plt.show();
    plt.close()


    train_hits = [train_accuracy]
    valid_hits = [validation_auc]
    
    test_hits = [test_auc, test_pos_accuracy, test_neg_accuracy]
    
    return train_hits, valid_hits, test_hits


def compute_auc(pos_score, neg_score):
    from sklearn.metrics import roc_auc_score
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)