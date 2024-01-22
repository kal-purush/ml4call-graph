from cmath import nan
from collections import Counter
import time
import dgl
import torch
from torch.utils.data import Dataset
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
from sklearn import preprocessing


def positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix(transpose=True,scipy_fmt='coo').astype(float)
    # A = g.adjacency_matrix().astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 
    return g

class NewDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.graph, train_pos_df, val_pos_df, test_pos_df, test_neg_df, val_neg_df= self.create_graph()  # single DGL graph
        self.train_edges = torch.from_numpy(train_pos_df.to_numpy())
        self.val_edges = torch.from_numpy(val_pos_df.to_numpy())  # positive val edges
        self.val_edges_neg = torch.from_numpy(val_neg_df.to_numpy())  # negative val edges
        self.test_edges = torch.from_numpy(test_pos_df.to_numpy())  # positive test edges
        self.test_edges_neg = torch.from_numpy(test_neg_df.to_numpy())  # negative test edges

    
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim):
        self.graph = positional_encoding(self.graph, pos_enc_dim)
    
    def get_function_edge(self, PATH):
        df = pd.read_csv(PATH)
        df = df.iloc[np.random.permutation(len(df))]
        left_out_edge_number = int(.9*len(df))
        train_df = df[:left_out_edge_number]
        leftout = df[left_out_edge_number:]
        leftout.to_csv("left_out_csv/leftout_"+self.name+".csv", index=False)
        return train_df
    
    def get_test_edge(self, ROOT_PATH, name):
        test_edges= self.get_function_edge(ROOT_PATH+ self.name+'_function_edges.csv')
        df1 = test_edges.iloc[np.random.permutation(len(test_edges))]
        train_len = int(.9*len(df1))
        train_func_df = df1[:train_len]
        # valid_len = int(.9*len(df1))
        # val_func_df = df1[train_len:valid_len]
        test_func_df = df1[train_len:]
        df = train_func_df
        test_df = test_func_df
        val_func_df = pd.DataFrame({})
        return df, test_df, val_func_df
        

    def create_graph(self, add_arg_len=True, add_param_len=True, add_name= True):
        import pandas as pd
        import torch.nn.functional as F
        import torch
        ROOT_PATH = "../prune_new/"
        df = pd.read_csv(ROOT_PATH+ self.name+'_node.csv')
        nodes_data = df.drop(["start_line","start_column","end_line",  "end_column" ,"file_name"], axis=1)
        dummy_nodes, dummy_edge = self.add_dummy_nodes(nodes_data, global_id=len(nodes_data), without_name_feature=False)
        print("New node ====> ", len(dummy_nodes), " New edge======> ", len(dummy_edge))
        nodes_data = pd.concat([nodes_data,dummy_nodes]).reset_index().drop(["index"], axis=1)

        df = pd.read_csv(ROOT_PATH+ self.name+'_edges.csv')

        suffled_df = df.iloc[np.random.permutation(len(df))].reset_index().drop(["index"], axis=1)
        train_len = int(.9*len(df))
        train_pos_df = suffled_df[:train_len]
        val_pos_df = suffled_df[train_len:]

        train_func_df, test_func_df, val_func_df = self.get_test_edge(ROOT_PATH, self.name)
        total_edges = len(train_func_df)+len(test_func_df)+len(val_func_df)
        
        df = pd.concat([df, dummy_edge]).reset_index().drop(["index"], axis=1)
        df_swapped = df.copy()
        df_swapped.columns = ['dst','src']
        df = pd.concat([df, df_swapped]).sort_index().reset_index(drop=True)

        a1D = np.array([0.1 for x in range(len(df))])
        t = torch.from_numpy(a1D)
        labels = torch.cat([t, torch.ones(2*total_edges)])

        
        function_df_conct = pd.concat([train_func_df, test_func_df, val_func_df]).reset_index().drop(["index"], axis=1)
        function_df_swapped = function_df_conct.copy()
        function_df_swapped.columns = ['dst','src']
        function_df = pd.concat([function_df_conct, function_df_swapped]).sort_index().reset_index(drop=True)

        edges_data = pd.concat([df,function_df]).reset_index().drop(["index"], axis=1)
        train_pos_df = pd.concat([train_pos_df,train_func_df]).reset_index().drop(["index"], axis=1)
        val_pos_df = pd.concat([val_pos_df,val_func_df]).reset_index().drop(["index"], axis=1)
        test_pos_df = test_func_df
        
        src = edges_data['src'].to_numpy()
        dst = edges_data['dst'].to_numpy()
        g = dgl.graph((src, dst))

        y= nodes_data['type'].tolist()
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(y)
        targets = torch.from_numpy(targets)
        targets = targets.type(torch.LongTensor)
        type_one_hot = F.one_hot(targets)
        print("y ==> ", len(y), len(type_one_hot))

        g.ndata['x_one_hot'] = type_one_hot
        y=torch.tensor(targets, dtype=torch.float32)
        # x=y.view(len(nodes_data),1)
        targets = torch.reshape(targets, (len(targets),1))
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
            name_one_hot = F.one_hot(targets)
            g.ndata['name_one_hot'] = name_one_hot
            targets = torch.reshape(targets, (len(targets),1))
            g.ndata['name'] = targets

        g.ndata['feat'] = torch.cat([g.ndata['x_one_hot'], g.ndata['args_len_one_hot'], g.ndata['param_len_one_hot'], g.ndata['name_one_hot']], dim=1)
        labels = labels.type(torch.LongTensor)
        labels = torch.reshape(labels, (len(labels),1))
        g.edata['feat'] = labels
        test_edges =  pd.concat([train_func_df, test_func_df, val_func_df]).reset_index().drop(["index"], axis=1)
        train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_df = self.create_neg_edge(g=g, nodes_data=nodes_data, test_edges=test_edges, train_size=len(train_pos_df), val_size=len(val_pos_df), test_size=len(test_pos_df))
        val_neg_df = {'src':val_neg_u,'v':val_neg_v}
        val_neg_df = pd.DataFrame.from_dict(val_neg_df)

        return g, train_pos_df, val_pos_df, test_pos_df, test_neg_df, val_neg_df
    
    


    def create_neg_edge(self, g, nodes_data, test_edges, train_size, val_size, test_size):
        
        all_neg_u, all_neg_v = dgl.sampling.global_uniform_negative_sampling(g, train_size+val_size)
        train_neg_u = all_neg_u[:train_size]
        train_neg_v = all_neg_v[:train_size]
        val_neg_u = all_neg_u[train_size:]
        val_neg_v = all_neg_v[train_size:]

        stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression']
        df1 = nodes_data[nodes_data.type.isin(stmt_type)]
        test_neg_id = df1['new_id'].tolist()
        
        df1 = nodes_data[((nodes_data['type']=='CallExpression')|(nodes_data['type']=='NewExpression'))]
        test_neg_call_site = df1['new_id'].tolist()
        
        test_edges.columns = ["src", "dst"]
        index = pd.MultiIndex.from_product([test_neg_call_site, test_neg_id], names = ["src", "dst"])
        all_combi = pd.DataFrame(index = index).reset_index()
        
        test_neg_df =  pd.merge(all_combi,test_edges, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        test_neg_df = test_neg_df.iloc[np.random.permutation(len(test_neg_df))]
        test_neg_df = test_neg_df[:test_size]
        
        return train_neg_u, train_neg_v, val_neg_u, val_neg_v, test_neg_df


    def add_dummy_nodes(self, nodes_data, global_id, without_name_feature=False):

        if without_name_feature:
            stmt_type=['Identifier','FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression', 'CallExpression']
        else:
            stmt_type=['Identifier']
        nodes_data = nodes_data[nodes_data.type.isin(stmt_type)]
        nodes_data = nodes_data.replace({'name': 'undefined'}, nan)
        nodes_data = nodes_data.dropna(subset=['name'])
        df_name_list = nodes_data['name'].tolist()
        
        temp_node_df = pd.DataFrame(columns=nodes_data.columns)
        temp_edge_df = pd.DataFrame(columns=['src','dst'], dtype='int64')
        for name in set(df_name_list):
            df2 = pd.DataFrame([[int(global_id), "special", name, -1, -1, int(global_id)]], columns=nodes_data.columns)
            temp_node_df = pd.concat([temp_node_df, df2], ignore_index=True)
            t_df = nodes_data[nodes_data['name']==name]
            ids = t_df['new_id'].tolist()

            for id_ in ids:
                df2 = pd.DataFrame([[int(id_), int(global_id)]], columns=['src','dst'],dtype='int64')
                temp_edge_df = pd.concat([temp_edge_df, df2], ignore_index=True)

            global_id+=1
            
        return temp_node_df, temp_edge_df