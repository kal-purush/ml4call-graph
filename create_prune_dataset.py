import glob
import os
import shlex
import subprocess
from time import sleep
import traceback
import pandas as pd
# project_list=['moment','commander.js', 'async', 'jszip', 'bluebird', 'chokidar']

excluded_types = ['LogicalExpression','BinaryExpression','ExpressionStatement', 'MemberExpression', 'BlockStatement', 'VariableDeclarator', 'AssignmentExpression', 'Property', 'ThisExpression'] 
# excluded_types =[]
included_types = ['CallExpression', 'Identifier', 'VariableDeclaration', 'FunctionExpression', 'FunctionDeclaration', 'ArrowFunctionExpression', 'Program' ]



from collections import defaultdict

class Graph:
    def __init__(self, V):
        # self.node_df = node_df
        self.V = V
        # print(V)
        self.graph = [[] for i in range(self.V)]
        self.prune_graph = [[] for i in range(self.V)]
        self.included_list=[]
        self.parent_dict={}
        self.parent_dict[0]=0
    
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.prune_graph[u].append(v)
        self.parent_dict[v]=u


    def BFS(self, s, type_dict):
        for node in range(1,len(self.prune_graph)):
            try:
                type_ = type_dict[node]
                if type_ in excluded_types:
                    parent = self.parent_dict[node]
                    # print(i, "====> ", parent)
                    self.prune_graph[parent].remove(node)
                    for child in self.prune_graph[node]:
                        if child not in self.prune_graph[parent]:
                            self.prune_graph[parent].append(child)
                        self.parent_dict[child]=parent
                    
            except:
                # print("error!")
                raise
                # pass
    
    def print_prune_graph(self, out_file):
        for node in range(len(self.prune_graph)):
            try:
                type_ = type_dict[node]
                if type_ not in excluded_types:
                    for child in self.prune_graph[node]:
                        out_file.write(str(node)+","+str(child)+"\n")
                    
            except:
                # print("error!")
                # raise
                pass
        out_file.flush()
    
    def create_node_file(self, node_df, out_file):
        df = node_df[~node_df['type'].isin(excluded_types)].reset_index().drop(["index"], axis=1)
        # print(df)
        df.to_csv(out_file,  index=False)

    
    def NumberOfconnectedComponents(self):
        # print("MAX ==> ", max(self.prune_graph)+1))
        visited = [False for i in range(self.V)]
        count = 0

        for v in range(len(self.prune_graph)):
            try:
                type_ = type_dict[v]
                if type_ not in excluded_types:
                    # print("key ====>  ", v);
                    if (visited[v] == False):
                        self.DFSUtil(v, visited)
                        print("returned")
                        count += 1
            except:
                pass
        return count
		
    def DFSUtil(self, v, visited):
        visited[v] = True
        for i in self.prune_graph[v]:
            if (not visited[i]):
                self.DFSUtil(i, visited)

def fix_ids(node_file, new_node_file, edge_file, new_edge_file, function_edge_file, new_function_edge_file):
    print("node file==> ", node_file)
    node_df = pd.read_csv(node_file)
    ids = node_df['id'].tolist()
    # print(node_df)
    new_ids=[]
    k=0
    id_dict={}
    for id in ids:
        id_dict[id]=k
        new_ids.append(k)
        k+=1
    print("K===> ", k)
    node_df['new_id'] = new_ids
    node_df.to_csv(new_node_file,  index=False)

    u=[]
    v=[]

    # print(id_dict)
    print(edge_file)
    with open(edge_file,"r+") as in_file:
        for line in in_file:
            if "src" not in line:
                parts = line.strip().split(",")
                src = id_dict[int(parts[0].strip())]
                dst = id_dict[int(parts[1].strip())]
                # print(src, dst)
                u.append(src)
                v.append(dst)
    
    new_df =pd.DataFrame({'src':u,'dst':v})
    new_df.to_csv(new_edge_file,  index=False)

    u=[]
    v=[]
    with open(function_edge_file,"r+") as in_file:
        for line in in_file:
            if "src" not in line:
                parts = line.strip().split(",")
                src = id_dict[int(parts[0].strip())]
                dst = id_dict[int(parts[1].strip())]
                # print(src, dst)
                u.append(src)
                v.append(dst)

    new_df =pd.DataFrame({'src':u,'dst':v})
    new_df.to_csv(new_function_edge_file,  index=False)

    

filepath = "../libraries/*"
folders = glob.glob(filepath)
# print(folders)
# project_list = ['lodash', 'express', 'formula-parser']
# project_list = ['Inquirer.js', 'underscore', 'angular', 'shelljs', 'js-yaml', 'commander.js', 'bluebird', 'through2', 'classnames', 
# 'body-parser', 'uuid', 'async', 'tslib', 'rxjs', 'core', 'package','q', 'node-fs-extra', 'request', 'minimist', 
# 'prop-types','debug', 'axios', 'colors.js','react']

# project_list = ['mathjs']
# project_list =  ['node-mongodb-native', 'mongoose', 'jest', 'aws-sdk-js', 'coffeescript', 'bootstrap', 'mocha', 'ramda', 'node-redis', 'webpack-dev-server', 'less.js', 'eslint-plugin-import', 'router', 'immutable-js']
project_list = ['moment', 'jsPDF', 'webpack', 'atompm', 'jquery', 'ws', 'cheerio', 'yargs', 'eslint-plugin-react', 'socket.io', 'sass-loader', 'handlebars.js']
project_list = ['cheerio', 'yargs', 'postcss', 'chai', 'superagent', 'eslint-plugin-jsx-a11y', 'node-fetch', 'chokidar', 'react-redux', 'autoprefixer', 'generator', 'html-webpack-plugin', 'winston', 'postcss-loader', 'qs', 'style-loader', 'ejs', 'node-sass']
project_list = ['cheerio', 'yargs', 'postcss', 'chai', 'superagent', 'eslint-plugin-jsx-a11y', 'node-fetch', 'chokidar', 'react-redux', 'autoprefixer', 'generator', 'html-webpack-plugin', 'winston', 'postcss-loader', 'qs', 'style-loader', 'ejs', 'node-sass', 'url-loader', 'node-semver', 'morgan', 'rimraf', 'node-glob', 'ember-cli-babel', 'promise', 'co']
project_list = ['mysql', 'joi','node-jsonwebtoken', 'create-react-app', 'react-router', 'UglifyJS']

# for name in folders:
#     index = name.rfind("/")
#     APPLICATION_NAME = name[index+1:]
#     # print(APPLICATION_NAME)
#     if APPLICATION_NAME in project_list:
#         print(APPLICATION_NAME)
#         try:
#             edge_out_file_name = "prune_ir/"+APPLICATION_NAME+"_edges.csv"
#             edge_out_file = open(edge_out_file_name, "w+")
#             node_out_file_name = "prune_ir/"+APPLICATION_NAME+"_nodes.csv"
#             node_out_file = open(node_out_file_name, "w+")
#             function_edge_file_name = "prune_ir/"+APPLICATION_NAME+"_function_edges.csv"

#             node_df = pd.read_csv("full_ast/"+APPLICATION_NAME+"_nodes.csv")
#             type_dict = dict(zip(node_df.id, node_df.type))

#             # for key in type_dict:
#             #     print(key, "====> ", type_dict[key])
#             #     break

#             g = Graph(len(node_df)+1)
#             with open("full_ast/"+APPLICATION_NAME+"_edges.csv", "r+") as in_file:
#                 for line in in_file:
#                     parts = line.strip().split(",")
#                     u= int(parts[0])
#                     v = int(parts[1])
#                     g.addEdge(u, v)

#             g.BFS(0, type_dict)
#             g.print_prune_graph(edge_out_file)
#             g.create_node_file(node_df, node_out_file)
#             # print(g.NumberOfconnectedComponents())
#             # print(node_df)

#             # fix_ids(node_out_file_name, new_node_file, edge_out_file_name, new_edge_file, function_edge_file_name, new_function_file )
#             # break
#         except:
#             traceback.print_exc()
#             # raise
#     #     # pass



# for name in project_list:
#     index = name.rfind("/")
#     APPLICATION_NAME = name[index+1:]
#     print(APPLICATION_NAME)

# sleep(30)

for APPLICATION_NAME in project_list:
    try:
        edge_out_file_name = "prune_ir/"+APPLICATION_NAME+"_edges.csv"
        node_out_file_name = "prune_ir/"+APPLICATION_NAME+"_nodes.csv"
        function_edge_file_name = "prune_ir/"+APPLICATION_NAME+"_function_edges.csv"

        new_node_file = "prune_new/"+APPLICATION_NAME+"_node.csv"
        new_edge_file = "prune_new/"+APPLICATION_NAME+"_edges.csv"
        new_function_file = "prune_new/"+APPLICATION_NAME+"_function_edges.csv"

        fix_ids(node_out_file_name, new_node_file, edge_out_file_name, new_edge_file, function_edge_file_name, new_function_file )
        # break
    except:
        traceback.print_exc()
        # raise





