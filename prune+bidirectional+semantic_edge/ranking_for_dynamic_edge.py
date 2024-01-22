"""
    IMPORTING LIBS
"""
import inspect
import sys
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

import random
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from NewDataset_for_ranking import NewDataset
import pandas as pd

import dgl.function as fn

# MODEL_NAME = '3WLGNN'
# MODEL_NAME = 'RingGNN'
# MODEL_NAME = 'MF'
# MODEL_NAME = 'MLP'
# MODEL_NAME = 'MLP'
# MODEL_NAME = 'GAT'
MODEL_NAME = 'GatedGCN'
# MODEL_NAME = 'GAT'
# MODEL_NAME = 'GraphSage'
# MODEL_NAME = 'DiffPool'
# MODEL_NAME = 'GCN'
    
# APPLICATION_NAME = 'mcstatic_0.0.20'
APPLICATION_NAME = 'is-my-json-valid_2.20.0'
DATASET_NAME = 'NEW_DATASET--'+APPLICATION_NAME

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

"""
    IMPORTING CUSTOM MODULES/METHODS
"""

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from nets.COLLAB_edge_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device
use_gpu = True; gpu_id = -1; device = None # CPU

def define_parameter(MODEL_NAME, dataset):
    
#     MODEL_NAME = 'MF'
    # MODEL_NAME = 'GatedGCN'
    
    n_heads = -1
    edge_feat = False
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    max_time = 12
    layer_type = 'dgl'
    num_embs = -1
    pos_enc = True
    #pos_enc = False
    pos_enc_dim = 10

    
    if MODEL_NAME == 'MF':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.01; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=0; hidden_dim=256; out_dim=hidden_dim; num_embs=235868;
    
    if MODEL_NAME == 'MLP':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=80; out_dim=hidden_dim; dropout=0.0; readout='mean'; gated = False  # Change gated = True for Gated MLP model
    
    if MODEL_NAME == 'GCN':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=5; hidden_dim=74; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'GraphSage':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=10; hidden_dim=38; out_dim=hidden_dim; dropout=0.0; readout='mean'; layer_type='edgefeat'

    if MODEL_NAME == 'GAT':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; n_heads=3; hidden_dim=19; out_dim=n_heads*hidden_dim; dropout=0.0; readout='mean'; layer_type='dgl'
    
    if MODEL_NAME == 'GIN':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=60; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'MoNet':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=53; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'GatedGCN':
        seed=41; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=5; hidden_dim=35; out_dim=hidden_dim; dropout=0.0; readout='mean'; edge_feat = False; layer_type='edgereprfeat'
        
    # generic new_params
    net_params = {}
    net_params['device'] = device
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    num_classes = 1
    net_params['n_classes'] = num_classes
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "mean"
    net_params['layer_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop
    net_params['layer_type'] = layer_type
    net_params['num_embs'] = num_embs
    net_params['gated'] = gated
    net_params['pseudo_dim_MoNet'] = 2
    net_params['kernel'] = 3
    net_params['n_mlp_GIN'] = 2
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'
    net_params['sage_aggregator'] = 'maxpool'   
    net_params['pos_enc'] = pos_enc
    net_params['pos_enc_dim'] = pos_enc_dim
    params = {}
    params['seed'] = seed
    params['epochs'] = epochs
    params['batch_size'] = batch_size
    params['init_lr'] = init_lr
    params['lr_reduce_factor'] = lr_reduce_factor 
    params['lr_schedule_patience'] = lr_schedule_patience
    params['min_lr'] = min_lr
    params['weight_decay'] = weight_decay
    params['print_epoch_interval'] = 5
    params['max_time'] = max_time
    return net_params, params


def view_model_param(MODEL_NAME, net_params):
    # print(net_params)
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    # print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    # print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


def set_parameters(MODEL_NAME, dataset, DATASET_NAME):
    net_params, params = define_parameter(MODEL_NAME=MODEL_NAME, dataset=dataset)
    config = {}
    gpu = {}
    gpu['use'] = use_gpu
    gpu['id'] = gpu_id
    config['gpu'] = gpu
    # GNN model, dataset, out_dir
    config['model'] = MODEL_NAME
    config['dataset'] = DATASET_NAME
    out_dir = 'out/debug/'
    config['out_dir'] = out_dir
    config['params'] = params
    # network parameters
    config['net_params'] = net_params
    params = config['params']
    DATASET_NAME = config['dataset']
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    out_dir = config['out_dir']
    MODEL_NAME = config['model']
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['n_classes'] = 1  # binary prediction
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    
    return net_params, params

def load_datset(DATASET_NAME):
    print("[I] Loading data (notebook) ...")
    parts = DATASET_NAME.split("--")
    APPLICATION_NAME = parts[1]
    # dataset = LoadData(DATASET_NAME)
    dataset = NewDataset(APPLICATION_NAME)
    print("[I] Finished loading.....")
    return dataset

def get_EPOC_Number(PATH):
    file_path = PATH+"/ROC_CURVE/*"
    project_list = glob.glob(file_path)
    max_ = -1
    for line in project_list:
        index = line.rfind("_")
        last_index = line.rfind(".png")
        number = int(line[index+1:last_index])
        max_ = max(max_, number)
    # print(max_)
    return max_

def train_model(dataset, EPOC_NUMBER, APPLICATION_NAME, net_params, params, MODEL_NAME):
    t0 = time.time()    
    DATASET_NAME = dataset.name

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            dataset._add_positional_encodings(net_params['pos_enc_dim'])

    graph = dataset.graph
    evaluator=""
    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg
    device = net_params['device']
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    out_dir = 'out/debug/'
    PATH = out_dir + 'checkpoints/' + MODEL_NAME + "_NEW_DATASET--" + APPLICATION_NAME + "/RUN_/epoch_"+str(EPOC_NUMBER)+".pkl"
    t = torch.load(PATH)
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=params['lr_reduce_factor'],
                                                        patience=params['lr_schedule_patience'],
                                                        verbose=True)

    return model, graph

import traceback
from collections import Counter
def filter_missed_edge(PATH, EPOC_NUMBER, model, graph, params, DATASET_NAME, MODEL_NAME, APPLICATION_NAME):
    missed_df = pd.read_csv("../missed_edge/"+APPLICATION_NAME+"_missed_call_site_ids.csv")
    ids = missed_df['id'].tolist()
    lst = ids
#     print(len(lst))
    node_df_org = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
    node_df = node_df_org[node_df_org['new_id'].isin(lst)].reset_index(drop=True)

    total = 0
    all_edges = []
    lst = []
    exclude_list=[]
    touple_lst = []
    include_list = [38024]
    for i in range(len(node_df)):
        try:
            file_name = node_df.iloc[i]['file_name']
            name = node_df.iloc[i]['name']
            start_line = node_df.iloc[i]['start_line']
            start_column = node_df.iloc[i]['start_column']
            src_id = node_df.iloc[i]['new_id']
            # if name not in exclude_list:
            if int(src_id) in include_list:
            # if name =='fn':
            # if pd.isnull(name):
                lst.append(name)
                # print(file_name, name, start_line, start_column, src_id)
                touple_lst.append((file_name, name, start_line, start_column, src_id))
                
        except:
            # traceback.print_stack()
            raise
    return touple_lst

import traceback

def get_test_edges(PATH, EPOC_NUMBER, model, graph, params, DATASET_NAME, MODEL_NAME, APPLICATION_NAME):
    df = pd.read_csv('../prune+bidirectional+semantic_edge+dynamic_edge/dynamic_edges/prune_dynamic_edges_'+APPLICATION_NAME+'.csv')
    df = df.drop_duplicates().reset_index(drop=True)
    ids = df['src'].tolist()
    lst = ids
    dst_dict = dict(df.values)

    node_df_org = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
    node_df = node_df_org[node_df_org['new_id'].isin(lst)].reset_index(drop=True)

    total = 0
    all_edges = []
    lst = []
    exclude_list=[]
    touple_lst = []
    include_list = [38024]
    for i in range(len(node_df)):
        try:
            file_name = node_df.iloc[i]['file_name']
            name = node_df.iloc[i]['name']
            start_line = node_df.iloc[i]['start_line']
            start_column = node_df.iloc[i]['start_column']
            src_id = node_df.iloc[i]['new_id']
            dst_id = dst_dict[src_id]
            touple_lst.append((file_name, name, start_line, start_column, src_id, dst_id))
                
        except:
            # traceback.print_stack()
            raise
    return touple_lst

def get_test_data(src_id, for_true_negative=False, APPLICATION_NAME=""):
    # print(src_id)
    ROOT_PATH = "../prune_new/"
    df = pd.read_csv(ROOT_PATH+ APPLICATION_NAME+'_node.csv')
    d = dict()
    f = open('../connected_files/'+APPLICATION_NAME+'_dependency_graph.json')
    data = json.load(f)
    u=[]
    v=[]
    temp_df_src = df[df['new_id']==src_id]
    file_name = temp_df_src.iloc[0]['file_name']
    new_lst = []
    new_lst.append(file_name)
    try:
        lst = data[file_name]
        for x in lst:
            if "lodash/internal" in x:
                x = x.replace("lodash/internal","lodash/.internal")
            new_lst.append(x)
    except:
        pass
    
    stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression']
    dst_df = df[(df.type.isin(stmt_type)) & (df.file_name.isin(new_lst))]
    if for_true_negative:
        return dst_df
    # print(dst_df['file_name'])
    # print(file_name, "====>",  len(dst_df))
    test_neg_id = dst_df['new_id'].tolist()
    done_list={}
    for node in test_neg_id:
        if (src_id,node) not in done_list:
                u.append(src_id)
                v.append(node)
                done_list[(src_id,node)]=True

    d={'src':u, 'dst':v}
    test_df = pd.DataFrame(d)
    test_edges = torch.from_numpy(test_df.to_numpy())
    return test_edges

def evaluate_network(model, device, graph, test_edges,
                     batch_size, DATASET_NAME="", MODEL_NAME="", SOURCE_NODE=0, DST_NODE=0, APPLICATION_NAME="", EPOC_NUMBER=0, TARGET=0):
    
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()
        try:
            x_pos_enc = graph.ndata['pos_enc'].to(device)
            h = model(graph, x, e, x_pos_enc) 
        except:
            h = model(graph, x, e)

        test_edges = test_edges.to(device)
        test_preds = []
        for perm in DataLoader(range(test_edges.size(0)), batch_size):
            edge = test_edges[perm].t()
            # print("edge ==> ",edge, len(edge))
            test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
#         print(test_preds)
        if len(test_edges)==1:
            t= test_preds[0]
            t =np.expand_dims(t,0)
            t = torch.tensor(t)
            test_preds = [t]
        test_pred = torch.cat(test_preds, dim=0)
    return test_edges, test_pred

        


def get_candidates(PATH, EPOC_NUMBER, model, graph, params, DATASET_NAME, MODEL_NAME, APPLICATION_NAME, SOURCE_ID, TARGET):
    node_df_org = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
    node_df = node_df_org[node_df_org['new_id']==SOURCE_ID].reset_index(drop=True)
    total = 0
    all_edges = []
    for i in range(len(node_df)):
        try:
            src_id = node_df.iloc[i]['new_id']
            test_edges = get_test_data(src_id=src_id, APPLICATION_NAME=APPLICATION_NAME)
            # print(test_edges)
            if len(test_edges) > 0:
                all_edges.append(test_edges)
        except:
            traceback.print_exc()
            pass
    
#     print(print("Total edge before ===> ",(all_edges)))
    all_edges = torch.cat(all_edges)
#     print("Total edge ===> ",(all_edges))
    if len(all_edges) > 0:
        test_edges, test_pred = evaluate_network(
                        model, device, graph, all_edges, params['batch_size'], DATASET_NAME=DATASET_NAME, MODEL_NAME=MODEL_NAME, SOURCE_NODE=src_id, APPLICATION_NAME=APPLICATION_NAME, TARGET=TARGET)
        # total+=count
    # print(all_edges)
    # print("Total edge ===> ",total)
    return test_edges, test_pred


def plot_figure(rank_lst):
    count_dict={}
    for i in range(0, 22):
        count_dict[i] = 0
    xticks_list=[]
    for x in rank_lst:
        key = x
        if x >20:
            key = 21
        count= count_dict[key]
        count_dict[key]=count+1
             
    for x in range(21):
        xticks_list.append(str(x))
    xticks_list.append('20+')
        
    count_dict = dict(sorted(count_dict.items()))

    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        plt.figure(figsize=(16, 8))
        categories = list(count_dict.keys())
        counts = list(count_dict.values())

        # Create a bar plot using Seaborn
        ax = sns.barplot(x=categories, y=counts)
        plt.xticks(range(len(xticks_list)), xticks_list)
        plt.title('Histogram of Candidate Ranking')
        plt.xlabel('Ranking')
        plt.ylabel('Count')
        plt.savefig("candidate_figures/dynamic_"+APPLICATION_NAME+".pdf")
        plt.close()
        plt.show()
    except:
        pass

MODEL_NAME = 'GatedGCN'
# to_do_list = ['formula-parser', 'lodash', 'js-yaml','express']
# app_list =  ['formula-parser', 'lodash', 'js-yaml','express']
app_list = ['winston', 'ws', 'qs', 'node-fs-extra', 'UglifyJS']

# to_do_list = ['mathjs']
# app_list =  ['mathjs']
for i in range(len(app_list)):
    try:
        APPLICATION_NAME = app_list[i]
        DATASET_NAME = 'NEW_DATASET--'+APPLICATION_NAME
        PATH = "results/NEW_DATASET--"+APPLICATION_NAME+"/"+MODEL_NAME
        EPOC_NUMBER = get_EPOC_Number(PATH=PATH)
        print(EPOC_NUMBER)
        dataset = load_datset(DATASET_NAME)
        net_params, params = set_parameters(MODEL_NAME, dataset, DATASET_NAME)
        model, graph = train_model(dataset, EPOC_NUMBER, APPLICATION_NAME, net_params, params, MODEL_NAME)
        touple_list = get_test_edges(PATH, EPOC_NUMBER, model, graph, params, DATASET_NAME, MODEL_NAME, APPLICATION_NAME)
        # print(touple_list)

        output_file = open("dynamic_"+APPLICATION_NAME+"_missing_edge_log_"+MODEL_NAME+".txt","w+")
        cutoff_thrshld = 0.9
        node_df_org = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
        
        count = 0
        rank_lst= []
        for i in range(len(touple_list)):
            file_name, name, start_line, start_column, SOURCE, TARGET = touple_list[i]
            test_edges, test_pred = get_candidates(PATH, EPOC_NUMBER, model, graph, params, DATASET_NAME, MODEL_NAME, APPLICATION_NAME, SOURCE, TARGET)
            output_file.write("Source =====>"+file_name+" "+str(name)+" "+str(start_line)+" "+str(start_column)+" "+str(SOURCE)+"\n")
            node_lst, pred_lst  =[], []
            for i in range(len(test_pred)):
                dst_id = test_edges[i][1].item()
                node_df = node_df_org[node_df_org['new_id']==dst_id].reset_index(drop=True)
                file_name = node_df.iloc[0]['file_name']
                name = node_df.iloc[0]['name']
                start_line = node_df.iloc[0]['start_line']
                start_column = node_df.iloc[0]['start_column']
                src_id = node_df.iloc[0]['new_id']
                pred_score = "{:.12f}".format(float(test_pred[i].item()))
                output_file.write(file_name+" "+str(name)+" "+str(start_line)+" "+str(start_column)+" "+str(src_id)+" =======> "+str(pred_score)+"\n")
                node_lst.append(src_id)
                pred_lst.append(pred_score)

            output_file.write("Total edge above "+str(cutoff_thrshld)+" ===>  "+str(count)+"\n")
            output_file.write("\n\n")    
            output_file.flush()

            result_df = pd.DataFrame(list(zip(node_lst, pred_lst)), columns =['id', 'val'])
            # print(result_df)
            result_df = result_df.sort_values(by=['val'], ascending=False).reset_index(drop=True)
            for i in range(len(result_df)):
                id = result_df.iloc[i]['id']
                if id == TARGET:
                    rank_lst.append(i)
        plot_figure(rank_lst)
    except:
        raise
