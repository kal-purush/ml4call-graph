import pandas as pd

to_do_list = ['formula-parser', 'lodash', 'mathjs','express','js-yaml']
app_list = ['formula-parser', 'lodash', 'mathjs','express','js-yaml']
to_do_list = ['js-yaml']
app_list = ['js-yaml']

# for APPLICATION_NAME in to_do_list:
#     df = pd.read_csv("../missed_edge/"+APPLICATION_NAME+".csv", header=None)
#     print(df)
#     node_df = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
#     print(node_df)
#     lst = ['CallExpression', 'NewExpression']
#     node_id_list = []
#     for i in range(len(df)):
#         file_name = df.iloc[i][3].split(":")
#         file_name = file_name[0]
#         # print(file_name)
#         start_line = int(df.iloc[i][5])
#         column_number = int(df.iloc[i][6])
#         end_line = int(df.iloc[i][7])
#         end_column_number = int(df.iloc[i][8])

#         temp_df = node_df[(node_df['file_name']== file_name) & (node_df['start_line']==start_line) & (node_df.type.isin(lst))].reset_index(drop=True)
#         # temp_df = node_df[(node_df['file_name']== file_name)]
#         if len(temp_df)==1:
#             id = temp_df.iloc[0]['new_id']
#             node_id_list.append(id)
#         elif len(temp_df)>1:
#             temp_df_ = temp_df[(temp_df['start_column']==column_number-1) & (temp_df['end_line']==end_line) & (temp_df['end_column']==end_column_number)].reset_index(drop=True)
#             if len(temp_df_)==1:
#                 id = temp_df_.iloc[0]['new_id']
#                 node_id_list.append(id)
#             elif len(temp_df_)==0:
#                 print(end_line, end_column_number)
#                 print(temp_df)
#         # else:
#         #     print(file_name, start_line, column_number)    

#     result_df = pd.DataFrame({})
#     result_df['id'] = node_id_list
#     result_df.to_csv("missed_edges/"+APPLICATION_NAME+"_missed_call_site_ids.csv", index=False)
    
for APPLICATION_NAME in to_do_list:
    print(APPLICATION_NAME)
    try:
        df = pd.read_csv("../csv_files/"+APPLICATION_NAME+"_iv_function.csv", header=None)
        print(len(df))
        node_df = pd.read_csv("../prune_new/"+APPLICATION_NAME+"_node.csv")
        # print(node_df)
        lst = ['CallExpression', 'NewExpression']
        node_id_list = []
        for i in range(len(df)):
            file_name = df.iloc[i][4].split(":")
            file_name = "/Users/masudulhasanmasudbhuiyan/Documents/gitlab/libraries/"+APPLICATION_NAME+file_name[0]
            print("file name ==> ", file_name)
            start_line = int(df.iloc[i][5])
            column_number = int(df.iloc[i][6])
            end_line = int(df.iloc[i][7])
            end_column_number = int(df.iloc[i][8])

            temp_df = node_df[(node_df['file_name']== file_name) & (node_df['start_line']==start_line) & (node_df.type.isin(lst))].reset_index(drop=True)
            # print(temp_df)
            # temp_df = node_df[(node_df['file_name']== file_name)]
            if len(temp_df)==1:
                id = temp_df.iloc[0]['new_id']
                node_id_list.append(id)
            elif len(temp_df)>1:
                temp_df_ = temp_df[(temp_df['start_column']==column_number-1) & (temp_df['end_line']==end_line) & (temp_df['end_column']==end_column_number)].reset_index(drop=True)
                if len(temp_df_)==1:
                    id = temp_df_.iloc[0]['new_id']
                    node_id_list.append(id)
                elif len(temp_df_)==0:
                    print(end_line, end_column_number)
                    print(temp_df)
            else:
                print(file_name, start_line, column_number)
    except:
        pass    

    result_df = pd.DataFrame({})
    result_df['id'] = node_id_list
    result_df.to_csv("id_files/"+APPLICATION_NAME+"_iv_function_ids.csv", index=False)