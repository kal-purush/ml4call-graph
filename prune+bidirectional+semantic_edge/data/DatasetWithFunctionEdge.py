from cgi import test
from cmath import nan
import time
import dgl
import torch
from torch.utils.data import Dataset

# from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from scipy import sparse as sp
import numpy as np
import random
import dgl
import torch
import torch.nn as nn

import itertools
import numpy as np
import scipy.sparse as sp

import pandas as pd

import dgl.function as fn

def positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 
    return g



class DatasetWithFunctionEdge(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        # self.dataset = DglLinkPropPredDataset(name='ogbl-collab')
        self.graph, train_pos_df, val_pos_df, test_pos_df, test_neg_df, val_neg_df= self.create_graph()  # single DGL graph

        import torch
        self.train_edges = torch.from_numpy(train_pos_df.to_numpy())
        self.val_edges = torch.from_numpy(val_pos_df.to_numpy())  # positive val edges
        self.val_edges_neg = torch.from_numpy(val_neg_df.to_numpy())  # negative val edges
        self.test_edges = torch.from_numpy(test_pos_df.to_numpy())  # positive test edges
        self.test_edges_neg = torch.from_numpy(test_neg_df.to_numpy())  # negative test edges

    
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph = positional_encoding(self.graph, pos_enc_dim)

    def create_graph(self, add_arg_len=True, add_param_len=True, add_name= True):
        import pandas as pd
        import torch.nn.functional as F
        import torch
        df = pd.read_csv('/Users/masudulhasanmasudbhuiyan/Music/test_ast_javascript/vue_nodes.csv')
        # print(df)
        nodes_data = df.drop(["start_line","start_column","end_line",  "end_column" ,"file_name"], axis=1)

        stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression', 'CallExpression']
        nodes_data = nodes_data[nodes_data.type.isin(stmt_type)]
        nodes_data = nodes_data.replace({'name': 'undefined'}, nan)
        ids = nodes_data['id'].tolist()

        id_dict={}
        k=0
        id_list = []
        for x in ids:   
            id_dict[x]=k
            id_list.append(k)
            k+=1
        
        with open("id_dictionary_for_dataset_with_function_edge.csv","w+") as out_file:
            for k in id_dict:
                out_file.write(str(k)+","+str(id_dict[k])+"\n")
                out_file.flush()
        
        nodes_data['id']=id_list
        # print(nodes_data)
        test_df= pd.read_csv('/Users/masudulhasanmasudbhuiyan/Music/test_ast_javascript/vue_function_edges.csv', header=None)
        test_edges = test_df.to_numpy()
        u,v=[],[]
        
        for i in range(len(test_edges)):
            src = id_dict[test_edges[i][0]]
            dst = id_dict[test_edges[i][1]]
            u.append(src)
            v.append(dst)

        d = {'src':u,'dst':v}
        df1 = pd.DataFrame(d)
        df1 = df1.iloc[np.random.permutation(len(df1))].reset_index().drop(["index"], axis=1)

        train_len = int(.8*len(u))
        train_func_df = df1[:train_len]
        valid_len = int(.9*len(u))
        val_func_df = df1[train_len:valid_len]
        test_func_df = df1[valid_len:]

        train_pos_df = train_func_df
        val_pos_df = val_func_df.reset_index().drop(["index"], axis=1)
        test_pos_df = test_func_df.reset_index().drop(["index"], axis=1)
        # print(test_pos_df)

        import dgl
        g = dgl.graph((u, v))
        # print(g)
        nodes_data = nodes_data[:g.num_nodes()]
        # print(nodes_data)

        y= nodes_data['type'].tolist()
        from sklearn import preprocessing
        import torch

        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(y)
        targets = torch.from_numpy(targets)
        targets = targets.type(torch.LongTensor)
        
        type_one_hot = F.one_hot(targets)

        g.ndata['x_one_hot'] = type_one_hot
        y=torch.tensor(targets, dtype=torch.float32)
        x=y.view(len(nodes_data),1)
        targets = torch.reshape(targets, (len(targets),1))
        # print(targets)
        g.ndata['x'] = targets

        if add_param_len:
            le = preprocessing.LabelEncoder()
            y=nodes_data['params_len'].tolist()
            targets = le.fit_transform(y)
            targets = torch.as_tensor(targets)
            targets = targets.type(torch.LongTensor)
            param_len_one_hot = F.one_hot(targets)
            g.ndata['param_len_one_hot'] = param_len_one_hot
            targets = torch.reshape(targets, (len(targets),1))
            g.ndata['param_len'] = targets

        if add_arg_len:
            le = preprocessing.LabelEncoder()
            y=nodes_data['argument_len'].tolist()
            targets = le.fit_transform(y)
            targets = torch.as_tensor(targets)
            targets = targets.type(torch.LongTensor)
            args_len_one_hot = F.one_hot(targets)
            g.ndata['args_len_one_hot'] = args_len_one_hot
            targets = torch.reshape(targets, (len(targets),1))
            g.ndata['args_len'] = targets

        if add_name:
            le = preprocessing.LabelEncoder()
            y=nodes_data['name'].tolist()
            targets = le.fit_transform(y)
            targets = torch.as_tensor(targets)
            targets = targets.type(torch.LongTensor)
            targets = torch.reshape(targets, (len(targets),1))
            g.ndata['name'] = targets

        g.ndata['feat'] = torch.cat( 
            [g.ndata['x'], g.ndata['args_len'], g.ndata['param_len'], g.ndata['name']], dim=1
        )
        
        labels = torch.ones(len(test_edges))
        labels = labels.type(torch.LongTensor)
        labels = torch.reshape(labels, (len(labels),1))
        g.edata['feat'] = labels

        print(g)
        train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_df = \
            self.create_neg_edge(g=g, nodes_data=nodes_data, test_edges=test_df, train_size=len(train_pos_df), val_size=len(val_pos_df), test_size=len(test_pos_df))
        val_neg_df = {'src':val_neg_u,'v':val_neg_v}
        val_neg_df = pd.DataFrame.from_dict(val_neg_df)
        # print("==============>")
        # print(val_neg_df)
        return g, train_pos_df, val_pos_df, test_pos_df, test_neg_df, val_neg_df

    def create_neg_edge(self, g, nodes_data, test_edges, train_size, val_size, test_size):
        u, v = g.edges()
        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
        print(adj.shape)
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        
        stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression']
        df1 = nodes_data[nodes_data.type.isin(stmt_type)]
        test_neg_id = df1['id'].tolist()

        # print("func site ===>>>", len(test_neg_id))

        df1 = nodes_data[nodes_data['type']=='CallExpression']
        test_neg_call_site = df1['id'].tolist()

        # print("call site ===>>>", len(test_neg_call_site))

        test_edges.columns = ["src", "dst"]
        index = pd.MultiIndex.from_product([test_neg_call_site, test_neg_id], names = ["src", "dst"])
        all_combi = pd.DataFrame(index = index).reset_index()
        
        test_neg_df =  pd.merge(all_combi,test_edges, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        test_neg_df = test_neg_df.iloc[np.random.permutation(len(test_neg_df))]
        # print(test_neg_df)
        test_neg_df = test_neg_df[:test_size].reset_index().drop(["index"], axis=1)
        # print(test_neg_df)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        
        val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[:train_size]], neg_v[neg_eids[:train_size]]

        return train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_df