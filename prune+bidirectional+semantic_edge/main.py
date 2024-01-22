"""
    IMPORTING LIBS
"""
from collections import Counter
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
# """
#     USER CONTROLS
# """
def load_datset():
    
    root_path ="results/"
    try: 
        path = root_path+DATASET_NAME
        os.mkdir(path) 
    except OSError as error: 
        # print(error)
        pass
    
    try: 
        path = root_path+DATASET_NAME+"/"+MODEL_NAME
        os.mkdir(path) 
    except OSError as error: 
        pass

    try: 
        path = root_path+DATASET_NAME+"/"+MODEL_NAME+"/ROC_CURVE"
        os.mkdir(path) 
    except OSError as error: 
        pass

    try: 
        path = root_path+DATASET_NAME+"/"+MODEL_NAME+"/POS_PRED"
        os.mkdir(path) 
    except OSError as error: 
        pass

    try: 
        path = root_path+DATASET_NAME+"/"+MODEL_NAME+"/NEG_PRED"
        os.mkdir(path) 
    except OSError as error: 
        pass
    
    try: 
        path = root_path+DATASET_NAME+"/"+MODEL_NAME+"/THRESHOLD"
        os.mkdir(path) 
    except OSError as error: 
        pass


    out_dir = 'out/debug/'
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')

    print("[I] Loading data (notebook) ...")
    dataset = LoadData(DATASET_NAME)
    print("[I] Finished loading.")
    # print(dataset)

    return dataset


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
    
    # for MF
    net_params['num_embs'] = num_embs
    
    # for MLPNet 
    net_params['gated'] = gated
    
    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = 2
    net_params['kernel'] = 3
    
    # specific for GIN
    net_params['n_mlp_GIN'] = 2
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'
    
    # specific for graphsage
    net_params['sage_aggregator'] = 'maxpool'   
    
    # specific for pos_enc_dim
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
    

"""
    TRAINING CODE
"""

def get_function_edge(PATH):
        df = pd.read_csv(PATH)
        d = Counter(df['dst'])
        count = 0
        lst=[]
        for k in d:
            if d[k]==1:
                lst.append(k)

        d_ = df[~df['dst'].isin(lst)]
        return d_

def prepare_neg_edge(test_size, test_size_neg):
    ROOT_PATH = "../prune_new/"
    df = pd.read_csv(ROOT_PATH+ APPLICATION_NAME+'_node.csv')
    nodes_data = df.drop(["start_line","start_column","end_line",  "end_column" ,"file_name"], axis=1)
    
    stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression']
    df1 = nodes_data[nodes_data.type.isin(stmt_type)]
    test_neg_id = df1['new_id'].tolist()

    df1 = nodes_data[((nodes_data['type']=='CallExpression')|(nodes_data['type']=='NewExpression'))]
    test_neg_call_site = df1['new_id'].tolist()

    # test_edges= pd.read_csv(ROOT_PATH+ APPLICATION_NAME +'_function_edges.csv')
    test_edges= get_function_edge(ROOT_PATH+ APPLICATION_NAME +'_function_edges.csv')

    dynamic_edges = pd.read_csv("dynamic_edges/dynamic_edges_"+APPLICATION_NAME+".csv")
    dynamic_edges = dynamic_edges.drop_duplicates(keep='first')

    dynamic_edges = dynamic_edges.merge(test_edges, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    test_edges = pd.concat([test_edges, dynamic_edges])

    test_edges.columns = ["src", "dst"]
    index = pd.MultiIndex.from_product([test_neg_call_site, test_neg_id], names = ["src", "dst"])
    all_combi = pd.DataFrame(index = index).reset_index()
    
    test_neg_df =  pd.merge(all_combi,test_edges, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    test_neg_df = test_neg_df.iloc[np.random.permutation(len(test_neg_df))]
    # print(test_neg_df)
    test_neg_df = test_neg_df[:test_size]
    # print(test_neg_df)
    return test_neg_df


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding",net_params['pos_enc_dim'])
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-t0)
        
    graph = dataset.graph
    
    # evaluator = dataset.evaluator
    evaluator=""
    
    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""\
                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    # writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Graph: ", graph)
    print("Training Edges: ", len(train_edges))
    print("Validation Edges: ", len(val_edges) + len(val_edges_neg))
    print("Test Edges: ", len(test_edges) + len(test_edges_neg))

    print(net_params)
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits = [], [] 
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        raise NotImplementedError # gave OOM while preparing dense tensor
    else:
        from train.train_COLLAB_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
    
    # train_neg_df = prepare_neg_edge(len(train_edges), len(test_edges))
    train_neg_df = pd.DataFrame({})

    try:
        monet_pseudo = None
        if MODEL_NAME == "MoNet":
            print("\nPre-computing MoNet pseudo-edges")
            # for MoNet: computing the 'pseudo' named tensor which depends on node degrees
            us, vs = graph.edges()
            # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
            monet_pseudo = [ 
                [1/np.sqrt(graph.in_degree(us[i])+1), 1/np.sqrt(graph.in_degree(vs[i])+1)] 
                    for i in range(graph.number_of_edges())
            ]
            monet_pseudo = torch.Tensor(monet_pseudo)
        
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)    

                start = time.time()
                    
                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges, params['batch_size'], epoch, train_neg_df, monet_pseudo)
                
                epoch_train_hit, epoch_val_hit, epoch_test_hit = evaluate_network(
                    model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo, DATASET_NAME=DATASET_NAME,MODEL_NAME=MODEL_NAME)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_train_hits.append(epoch_train_hit)
                epoch_val_hits.append(epoch_val_hit)

                with open(write_file_name + '.txt', 'a+') as f:
                    f.write("""TEST AUC: {:.4f} TEST TP: {:.4f} TEST TN: {:.4f}\n"""\
                    .format(epoch_test_hit[0]*100, epoch_test_hit[1]*100, epoch_test_hit[2]*100))

                

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, train_hits=epoch_train_hits[0], 
                              val_hits=epoch_val_hits[0], test_hits=epoch_test_hit[0]) 

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss) 

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                # files = glob.glob(ckpt_dir + '/*.pkl')
                # for file in files:
                #     epoch_nb = file.split('_')[-1]
                #     epoch_nb = int(epoch_nb.split('.')[0])
                #     if epoch_nb < epoch-1:
                #         os.remove(file)

                scheduler.step(epoch_val_hit[0])

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    import matplotlib.pyplot as plt

    plt.plot(epoch_train_losses)
    root_path ="results/"
    
    path = root_path+'NEW_DATASET--'+DATASET_NAME+"/"+MODEL_NAME
    print(path)
    plt.savefig(path+'/train_loss',dpi=300)

    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))


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

tim1 = time.time()

# print("total time ==> ", time.time()-tim1)

def main(notebook_mode=False,config=None):    
    params = config['params']
    DATASET_NAME = config['dataset']
    # dataset = load_datset()
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
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

# to_do_list = ['formula-parser', 'lodash', 'express']
# to_do_list = ['Inquirer.js', 'underscore', 'angular', 'shelljs', 'js-yaml', 'commander.js', 'bluebird', 'through2', 'classnames', 
# 'body-parser', 'uuid', 'async', 'tslib', 'rxjs', 'core', 'package','q', 'node-fs-extra', 'request', 'minimist', 
# 'prop-types','debug', 'axios', 'colors.js','react']

# to_do_list =  ['node-mongodb-native', 'mongoose', 'jest', 'aws-sdk-js', 'coffeescript', 'bootstrap', 'mocha', 'ramda', 'node-redis', 'webpack-dev-server', 
#                'less.js', 'eslint-plugin-import', 'router', 'immutable-js']

# to_do_list =  ['jest', 'coffeescript', 'bootstrap', 'mocha', 'ramda', 'webpack-dev-server']
# to_do_list = ['chai', 'autoprefixer', 'winston', 'node-semver', 'html-webpack-plugin', 'qs', 'chokidar', 'postcss', 'postcss-loader', 'ejs', 'morgan', 'url-loader']
to_do_list = ['mysql', 'joi','node-jsonwebtoken', 'create-react-app', 'react-router', 'UglifyJS']
# to_do_list =  ['eslint-plugin-import']

filepath = "../prune_new/*"
project_list = glob.glob(filepath)
# print(project_list)
for file_ in project_list:
    try:
        if "_node" in file_:
            file_name = file_.split("/")[2]
            index = file_name.rfind("_node")
            APPLICATION_NAME = file_name[:index]
            print(APPLICATION_NAME)
            if APPLICATION_NAME in to_do_list:
                DATASET_NAME = 'NEW_DATASET--'+APPLICATION_NAME
                dataset = load_datset()
                net_params, params = define_parameter(MODEL_NAME=MODEL_NAME, dataset=dataset)
                config = {}
                gpu = {}
                gpu['use'] = use_gpu
                gpu['id'] = gpu_id
                config['gpu'] = gpu
                config['model'] = MODEL_NAME
                config['dataset'] = DATASET_NAME
                out_dir = 'out/debug/'
                config['out_dir'] = out_dir
                config['params'] = params
                # network parameters
                config['net_params'] = net_params
                main(True,config)
    
    except:
        # pass
        raise






    