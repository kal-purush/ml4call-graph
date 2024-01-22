const fs = require("fs");
const esprima = require("esprima");
const estraverse = require("estraverse");
const escodegen = require("escodegen");
const path = require("path");
const { Literal } = require("esprima");
const l = console.log
const clc = require('cli-color');
const findImports = require('find-imports');


var global_id = 0;
var global_dict = Object.create(null);

var dict_for_ident = Object.create(null);

var APPLICATION_NAME = "moment"
node_file_path = "csv_files/with_full_ast/"+APPLICATION_NAME+"_nodes_with_only_childern_node.csv";
edge_file_path = "csv_files/with_full_ast/"+APPLICATION_NAME+"_edges_with_only_childern_node.csv";
function_edge_file = "csv_files/with_full_ast/"+APPLICATION_NAME+"_function_edges.csv";
var done_once= false

// if (fs.existsSync(node_file_path)) {
//     fs.unlinkSync(node_file_path);
// }
// if (fs.existsSync(edge_file_path)) {
//     fs.unlinkSync(edge_file_path);
// }

// if (fs.existsSync(function_edge_file)) {
//     fs.unlinkSync(function_edge_file);
// }


function produce_dataset(ast, file_name, import_dict) {
    let done = false;
    estraverse.traverse(ast, {
      enter: function (node, parent) {
        
        var argument_len=-1, params_len=-1, name="", type;
        type = node.type;
        if (node.type == 'Literal')return;
        if(node.type == 'FunctionDeclaration' || node.type=='ArrowFunctionExpression' || node.type=='FunctionExpression' || node.type == 'CallExpression'|| node.type == 'NewExpression'){
            if(node.type == 'CallExpression' || node.type == 'NewExpression'){
                argument_len = node.arguments.length
                if(node.callee.name)name = node.callee.name
                else if(node.callee.property){
                    if(node.callee.property.name)name = node.callee.property.name
                }
                else if(node.callee.type == 'SequenceExpression'){
                    if(node.callee.expressions){
                        for(let item of node.callee.expressions){
                            if(item.type=='MemberExpression'){
                                if(item.property){
                                    if(item.property.name)name = item.property.name;
                                
                                    else if (item.object){
                                        if(item.object.name)name = item.object.name;
                                    }
                                }
                                else if (item.object){
                                    if(item.object.name)name = item.object.name;
                                }
                            }
                        }
                    }
                }
            }
            else if(node.type == 'FunctionDeclaration' || node.type=='ArrowFunctionExpression' || node.type=='FunctionExpression'){
                params_len = node.params.length
                if (node.id){
                    name = node.id.name
                }
            }
            else{
                if(node.name){
                    name = node.name;
                }
            }
        }
        else{
            if(node.name){
                name = node.name;
            }
        }
        // l(node, parent)
        node_key = type+file_name+":"+node.loc.start.line+"_"+node.loc.start.column+":"+node.loc.end.line+"_"+node.loc.end.column
        if(parent)parent_key = parent.type+file_name+":"+parent.loc.start.line+"_"+parent.loc.start.column+":"+parent.loc.end.line+"_"+parent.loc.end.column
        
        row = global_id+","+type+","+name+","+params_len+","+argument_len+",";
        row+=node.loc.start.line+","+node.loc.start.column+","+node.loc.end.line+","+node.loc.end.column+","+file_name+"\n";
        if(type=="Program"){
            l(file_name);
            global_dict[node_key]=global_id;
            fs.appendFileSync(node_file_path, row);
            global_id+=1;
        }

        else if(type=="Identifier"){
            var ident_id;
            identifier_key = name+file_name
            // if(name=='months')l(identifier_key)
            if (identifier_key in dict_for_ident){
                ident_id = dict_for_ident[identifier_key];
                // if(name=='months')l("found ===> ", identifier_key, file_name, ident_id);
                global_dict[node_key]=ident_id;
            }
            else{
                fs.appendFileSync(node_file_path, row);
                global_dict[node_key]=global_id;
                dict_for_ident[identifier_key]=global_id;     
                ident_id = global_id;
                global_id+=1;
                // if(name=='months')l("first time ===> ", identifier_key, file_name, ident_id);
            }

            if(parent){
                parent_id = global_dict[parent_key]
                row = parent_id+","+ident_id+"\n";
                // if(!row.includes("function"))
                fs.appendFileSync(edge_file_path, row);
            }
        }

        else{
            if(parent){
                fs.appendFileSync(node_file_path, row);

                if (node_key == parent_key){
                    // l("node ===> ", node);
                    // l(node_key)
                    // l("parent ===> ", parent)
                    global_dict[node_key]=0;
                }
                else{
                    parent_id = global_dict[parent_key]
                    row = parent_id+","+global_id+"\n";
                    fs.appendFileSync(edge_file_path, row);
                    global_dict[node_key]=global_id;
                }
                global_id+=1;
            }
        }
      },
    });
    // l(global_id)
    return ast;
}

let main_lst=[]

function get_file(file_path){
    // l(file_path);
    if (fs.lstatSync(path.join(file_path)).isDirectory()){
        var files = fs.readdirSync(file_path);
        for (const folder_path of files) {
            get_file(path.join(file_path, folder_path))
        }
    }
    else {
        if (file_path.endsWith(".js")){
            main_lst.push(file_path);
            // l(main_lst)
        }
    }

    // return main_lst;
}

function delete_files(){
    if (fs.existsSync(node_file_path)) {
        fs.unlinkSync(node_file_path);
    }
    if (fs.existsSync(edge_file_path)) {
        fs.unlinkSync(edge_file_path);
    }
    
    if (fs.existsSync(function_edge_file)) {
        fs.unlinkSync(function_edge_file);
    }
}

function get_file_name(relative_path, file_path){
    parts = file_path.split("/")
    // l(parts);
    let count = 0;
    first_back_slash=false
    let regex = /\.\.\//g;
    var found = relative_path.match(regex);
    if(file_path=="/Users/masudulhasanmasudbhuiyan/Music/test_ast_javascript/vuln4js_repos/ms_0.7.0/locale/ar-ly.js")l("count ===>", found, relative_path);
    let number_of_backtrack = 0
    if(found)number_of_backtrack += found.length

    // regex = /^'\.\//g;
    // found = relative_path.match(regex);
    // if(found)number_of_backtrack += found.length
    // // l("count ===>", number_of_backtrack, relative_path);
    number_of_backtrack+=1;
    var full_path =""
    for(let i=0;i<parts.length-number_of_backtrack;i++){
        full_path+=parts[i]
        full_path+='/'
    }
    
    if(file_path=="/Users/masudulhasanmasudbhuiyan/Music/test_ast_javascript/vuln4js_repos/ms_0.7.0/locale/ar-ly.js")l("number ==>", number_of_backtrack)

    regex = /\w.*'/g;
    found = relative_path.match(regex);
    return full_path+found[0].substring(0, found[0].indexOf('\''))+".js"
}

function get_imports(file_list){
    // l(file_list)
    var import_dict = {};

    for(let file_ of file_list){
        l("file =====> ", file_);
        const content = fs.readFileSync(file_).toString();
        let parts = content.split("\n")
        let count = 0;
        imported_file_lst=[]
        if (!file_.includes(".min.js") && !file_.includes("node_modules")){
            for(let line of parts){
                try{
                    line = line.replace('"', '\'')
                    let regex = /import.*{*.*}*.*from/g;
                    var found = line.match(regex);
                    if(found){
                        // l(found)
                        index = line.indexOf("from")
                        imported_file_name = line.substring(index+5,)
                        // l(imported_file_name, line, file_)
                        var full_path = get_file_name(imported_file_name, file_);
                        full_path = full_path.replace(".js.js",".js");
                        imported_file_lst.push(full_path);
                    }else{
                        regex = /from '.*';/g;
                        found = line.match(regex);
                        if(found){
                            // l(found)
                            index = line.indexOf("from")
                            imported_file_name = line.substring(index+5,)
                            // l(imported_file_name)
                            var full_path = get_file_name(imported_file_name, file_);
                            full_path = full_path.replace(".js.js",".js");
                            imported_file_lst.push(full_path);
                        }
                    }

                    if(line.includes("require(")){
                        index = line.indexOf("require(")
                        // l("index ===> ", index);
                        imported_file_name = line.substring(index+8,line.indexOf(')'))
                        // l(imported_file_name)
                        var full_path = get_file_name(imported_file_name, file_);
                        full_path = full_path.replace(".js.js",".js");
                        imported_file_lst.push(full_path);
                    }
                }catch(e){
                    // l("error ====> ", line, e)
                }
            }
        }
        // break
        // if(count>0)l(file_," =====> " ,count) 
        // l(imported_file_lst);
        import_dict[file_] = imported_file_lst;
    }
    // l(import_dict);
    // import_group_dict={}
    // var index_number = 0;
    // Object.entries(import_dict).forEach(([k,v]) => {
    //     if(k in import_group_dict){
    //         key_index = import_group_dict[k];
    //     }
    //     else{
    //         key_index = index_number;
    //         import_group_dict[k] = key_index;
    //         index_number+=1;
    //         l(k, key_index);
    //     }
    //     // console.log("The key: ", k)
    //     // console.log("The value: ", v)
    //     for(let lib_ of v){
    //         import_group_dict[lib_] = key_index;
    //     }
    // })

    // Object.entries(import_dict).forEach(([k,v]) => {
    //     console.log("The key: ", k)
    //     console.log("The value: ", v)
    // })

    // l(import_group_dict);
    return import_dict;
}

const storeData = (data, json_file_path) => {
    try {
        fs.writeFileSync(json_file_path, JSON.stringify(data, null, 4))
    } catch (err) {
        console.error(err)
    }
}

// var project_list =['ms_0.7.0','is-my-json-valid_2.20.0']
// var project_list =['js-data_3.0.9', 'atropa-ide_0.2.2', 'mathjs_3.9.0','mixin-pro_0.6.0', 'locutus_2.0.10' ]
// var project_list =['axios']
// var project_list = ['node-mkdirp', 'core', 'tslib', 'classnames', 'async', 'minimist', 'underscore', 'chalk', 'node-semver', 'generator', 'webpack', 'bluebird', 'through2', 'Inquirer.js', 'body-parser', 'colors.js', 'node-glob', 'node-fs-extra', 'commander.js', 'formula-parser_old', 'prop-types', 'request', 'yargs', 'rxjs', 'babel-runtime', 'uuid', 'react', 'package', 'debug']

// var project_list = ['cheerio', 'rimraf', 'q', 'shelljs', 'dotenv', 'angular', 'js-yaml', 'style-loader', 'winston', 'object-assign']
// var project_list =  ['node-mongodb-native', 'mongoose', 'jest', 'aws-sdk-js', 'coffeescript', 'bootstrap', 'mocha', 'ramda', 'node-redis', 'webpack-dev-server', 'less.js', 
// 'eslint-plugin-import', 'router', 'immutable-js']

// var project_list = ['moment', 'jsPDF', 'webpack', 'atompm', 'jquery', 'ws', 'cheerio', 'yargs', 'eslint-plugin-react', 'socket.io', 'sass-loader', 'handlebars.js']
var project_list = ['cheerio', 'yargs', 'postcss', 'chai', 'superagent', 'eslint-plugin-jsx-a11y', 'node-fetch', 'chokidar', 'react-redux', 'autoprefixer', 'generator', 'html-webpack-plugin', 'winston', 'postcss-loader', 'qs', 'style-loader', 'ejs', 'node-sass', 'url-loader', 'node-semver', 'morgan', 'rimraf', 'node-glob', 'ember-cli-babel', 'promise', 'co']
var project_list = ['mysql', 'joi','node-jsonwebtoken', 'create-react-app', 'react-router', 'UglifyJS']
try {
    let filepath = "/Users/masudulhasanmasudbhuiyan/Documents/gitlab/libraries";
    var files = fs.readdirSync(filepath);
    
    for(let file_ of files){
        // l(file_)
        if(project_list.includes(file_)){
            node_file_path = "full_ast/"+file_+"_nodes.csv";
            edge_file_path = "full_ast/"+file_+"_edges.csv";
            
            if (fs.existsSync(node_file_path)) {
                fs.unlinkSync(node_file_path);
            }
            if (fs.existsSync(edge_file_path)) {
                fs.unlinkSync(edge_file_path);
            }
            
            let file_path = "/Users/masudulhasanmasudbhuiyan/Documents/gitlab/libraries/"+file_;
            main_lst=[]
            global_id = 1;
            global_dict = Object.create(null);
            dict_for_ident = Object.create(null);
            get_file(file_path, main_lst);
            console.log("Get File Done ====>")
            // var import_dict = get_imports(main_lst);
            // var import_dict = {};
            console.log("Import Dict Done ====>")
            // delete_files();
            json_file_path =  "full_ast/"+file_+"_import_dict.json";
            // storeData(import_dict, json_file_path);

            row="id,type,name,params_len,argument_len,start_line,start_column,end_line,end_column,file_name\n"
            // // row+= "0,Program,,-1,-1,";
            // // row+=",,,,\n";
            fs.appendFileSync(node_file_path, row);
            for(let file_path of main_lst){
                if(!file_path.includes('node_modules') && !file_path.includes('babel_instrumentor.js')){
                    try {
                        var data = fs.readFileSync(file_path).toString();
                        // l(data)
                        var ast = esprima.parseModule(data, { comment: false, loc: true  });
                        produce_dataset(ast, file_path);
                    } catch (e) {
                        l(file_path)
                        l(e)
                        // throw Error(e)
                    }
                }
            }
        }
    }
// }


} catch (e) {
    throw Error(e);
}

// try {
//     let filepath = "/Users/masudulhasanmasudbhuiyan/Music/test_ast_javascript/gitrepos/"+APPLICATION_NAME;
//     get_file(filepath);
// } catch (e) {
//     throw Error(e);
// }




// row="id,type,name,params_len,argument_len,start_line,start_column,end_line,end_column,file_name\n"
// row+= "0,Program,,-1,-1,";
// row+=",,,,\n";
// fs.appendFileSync(node_file_path, row);
// for(let file_path of main_lst){
//     if(!file_path.includes("min.js") && !file_path.includes("test/")){
//         try {
//             var data = fs.readFileSync(file_path).toString();
//             // l(data)
//             var ast = esprima.parseModule(data, { comment: false, loc: true  });
//             produce_dataset(ast, file_path);
//         } catch (e) {
//             // l(file_path)
//             // l(e)
//             // throw Error(e)
//         }
//     }
    
// }

// add_function_edge_try();