
import glob
import os
import shlex
import subprocess
import time
import traceback
import pandas as pd
project_list=['.DS_Store']
# project_list= ['ms_0.7.0', 'is-my-json-valid_2.20.0']
# project_list =['js-data_3.0.9', 'atropa-ide_0.2.2-2', 'mathjs_3.9.0','mixin-pro_0.6.0', 'locutus_2.0.10' ]
# project_list = ['axios','express']
# project_list = ['node-mkdirp', 'core', 'tslib', 'classnames', 'async', 'minimist', 'underscore', 'chalk', 'node-semver', 'generator', 'webpack', 'bluebird', 'through2', 'Inquirer.js', 'body-parser', 'colors.js', 'node-glob', 'node-fs-extra', 'commander.js', 'formula-parser_old', 'prop-types', 'request', 'yargs', 'rxjs', 'babel-runtime', 'uuid', 'react', 'package', 'debug']
# project_list = ['cheerio', 'rimraf', 'q', 'shelljs', 'dotenv', 'angular', 'js-yaml', 'style-loader', 'winston', 'object-assign']
# project_list =  ['node-mongodb-native', 'mongoose', 'jest', 'aws-sdk-js', 'coffeescript', 'bootstrap', 'mocha', 'ramda', 'node-redis', 'webpack-dev-server', 'less.js', 'eslint-plugin-import', 'router', 'immutable-js']
# project_list = ['moment', 'jsPDF', 'webpack', 'atompm', 'jquery', 'ws', 'cheerio', 'yargs', 'eslint-plugin-react', 'socket.io', 'sass-loader', 'handlebars.js']
project_list = ['cheerio', 'yargs', 'postcss', 'chai', 'superagent', 'eslint-plugin-jsx-a11y', 'node-fetch', 'chokidar', 'react-redux', 
                'autoprefixer', 'generator', 'html-webpack-plugin', 'winston', 'postcss-loader', 'qs', 'style-loader', 'ejs', 'node-sass', 
                'url-loader', 'node-semver', 'morgan', 'rimraf', 'node-glob', 'ember-cli-babel', 'promise', 'co']

project_list = ['mysql', 'joi','node-jsonwebtoken', 'create-react-app', 'react-router', 'UglifyJS']

folders = glob.glob("/Users/masudulhasanmasudbhuiyan/Documents/gitlab/libraries/*")
# print(folders)

for folder in folders:
    # APPLICATION_NAME = "moment"
    index = folder.rfind("/")
    APPLICATION_NAME = folder[index+1:].strip()
    # print(APPLICATION_NAME)

    if APPLICATION_NAME in project_list:
        try:
            function_edge_file_name = "prune_ir/"+APPLICATION_NAME+"_function_edges.csv"
            function_edge_file = open(function_edge_file_name,"w+")
            # try:
            #     os.remove(function_edge_file)
            # except FileNotFoundError:
            #     pass

            function_edge_file.write("src, dst\n")

            node_df = pd.read_csv("full_ast/"+APPLICATION_NAME+"_nodes.csv")

            stmt_type=['FunctionDeclaration', 'ArrowFunctionExpression', 'FunctionExpression']
            caller_df = node_df[((node_df['type']=='CallExpression')|(node_df['type']=='NewExpression'))]
            callee_df = node_df[node_df['type'].isin(stmt_type)]

            caller_dict= {}
            callee_dict = {}

            time1 = time.time()

            for i in range(len(caller_df)):
                id = caller_df.iloc[i]['id']
                # print(caller_df.iloc[i])
                # print(caller_df.iloc[i]['file_name'])
                # print(caller_df.iloc[i]['start_line'])
                # print(caller_df.iloc[i]['start_column'])
                key = caller_df.iloc[i]['file_name']+'-'+ str(int(caller_df.iloc[i]['start_line']))+'-'+str(int(caller_df.iloc[i]['start_column']))
                # print(key)
                caller_dict[key]=id
            
            for i in range(len(callee_df)):
                id = callee_df.iloc[i]['id']
                key = callee_df.iloc[i]['file_name']+'-'+ str(int(callee_df.iloc[i]['start_line']))+'-'+ str(int(callee_df.iloc[i]['start_column']))
                callee_dict[key]=id

            # print("Total Running time = {:.3f} seconds".format(time.time() - time1))

            # time1 = time.time()

            actual_application_name = APPLICATION_NAME.split("_")[0]
            print(actual_application_name)

            df = pd.read_csv("/Users/masudulhasanmasudbhuiyan/Documents/gitlab/libraries/call_edges_from_codeql/"+actual_application_name+".csv", header=None)
            # print(df)
            count = 0
            for i in range(len(df)):
                # print(i)
                parts = df.iloc[i][3].split("\n")
                # print(parts)
                for part in parts:
            #         # if "test/" not in part and "data/externs" not in part:
                    if "data/externs" not in part:
                        split_parts = part.split("-------->")
                        src = split_parts[0]
                        dst = split_parts[1]
                        callee_id = -1
                        caller_id = -1
                        
                        src_part = src.split("->")
                        src_key = src_part[0]+":"+src_part[1]
                        end_index = src_part[0].rfind(":")
                        file_name = src_part[0][:end_index]
                        start_line = int(src_part[0][end_index+1:])
                        column_value = int(src_part[1])-1
                        key = file_name+"-"+str(start_line)+"-"+str(column_value)
                        key_alt = file_name+"-"+str(start_line-1)+"-"+str(column_value)
                        if key in caller_dict or key_alt in caller_dict: 
                            if key in caller_dict:
                                caller_id = caller_dict[key]
                            else:
                                caller_id = caller_dict[key_alt]
                        else:
                            print("not found", key)



                        dst_part = dst.split("->")
                        end_index = dst_part[0].rfind(":")
                        file_name = dst_part[0][:end_index]            
                        start_line = int(dst_part[0][end_index+1:])
                        column_value = int(dst_part[1])-1
                        key = file_name+"-"+str(start_line)+"-"+str(column_value)
                        key_alt = file_name+"-"+str(start_line-1)+"-"+str(column_value)
                        if key in callee_dict or key_alt in callee_dict: 
                            if key in callee_dict:
                                callee_id = callee_dict[key]
                            else:
                                calcallee_id = callee_dict[key_alt]
                        else:
                            print("callee not found", key)
                        
                        if callee_id!= -1 and  caller_id!= -1:
                            function_edge_file.write(str(caller_id)+","+str(callee_id)+"\n")
                            function_edge_file.flush()
                        
            print("Total Running time = {:.3f} seconds".format(time.time() - time1))
            print(count)
        except:
            traceback.print_exc()
            # pass