const fs = require("fs");
const esprima = require("esprima");
const estraverse = require("estraverse");
const escodegen = require("escodegen");
const path = require("path");
const { Literal } = require("esprima");
const l = console.log
const clc = require('cli-color');
const findImports = require('find-imports');
const { full } = require("acorn-walk");

var global_id = 0;
var global_dict = Object.create(null);
var dict_for_ident = Object.create(null);

function get_full_path(file_name, value, node){
    full_path = path.resolve(path.dirname(file_name), value);
    // console.log("path ===> ", full_path, value, path.dirname(file_name))
    if(!full_path.endsWith(".js")){
        full_js_path = full_path+".js"
    }else{
        full_js_path = full_path
    }
    if(value=='..'){
        js_file_name = node.specifiers[0].local.name;
        full_js_path = full_path+"/"+js_file_name+".js"
    }
    
    // if(value.includes("angular-base-package")| value.includes("examples-package")| value.includes("remark-package")| value.includes("target-package") | 
    //     value.includes("links-package")| value.includes("links-package")){
    //     // console.log("full_js_path ===> ", full_js_path)
    //     full_js_path = full_path+"/index.js"
    // }

    // console.log("full_js_path ===> ", full_js_path)
    if (!fs.existsSync(full_js_path)) {
        // console.log("full_js_path ===> ", full_js_path)
        // console.log("file not found ===> ", full_js_path, file_name, value)
        full_js_path = full_path+"/index.js"
        if (!fs.existsSync(full_js_path)) {
            console.log("file not found ===> ", full_js_path, file_name, value)
        }
        else return full_js_path;
    }
    return full_js_path;
}

function produce_dataset(ast, file_name, import_dict) {
    let done = false;
    const dependencies = [];
    estraverse.traverse(ast, {
      enter: function (node, parent) {
        
        var argument_len=-1, params_len=-1, name="", type;
        type = node.type;
        if (node.type == 'Literal')return;
        if(node.type == 'ImportDeclaration'){
            value = node.source.value
            // console.log("value ===> ", value)
            if(value.startsWith(".")){
                try{
                    dependencies.push(get_full_path(file_name, value, node));
                }catch(e){
                    console.log("error ===>", e)
                }
            }
        }
        else if(node.type == 'VariableDeclaration'){
            // const requireStatements = [];
            node.declarations.forEach(declaration => {
                if (
                  declaration.init &&
                  declaration.init.type === 'CallExpression' &&
                  declaration.init.callee.name === 'require' &&
                  declaration.init.arguments.length === 1 &&
                  declaration.init.arguments[0].type === 'Literal'
                ) {
                    value = declaration.init.arguments[0].value;
                    if(value.startsWith(".")){
                        dependencies.push(get_full_path(file_name, value));
                    }
                }
              });
            // if(requireStatements.length>0)console.log("requireStatements ===> ", requireStatements)
        }

       
      },
    });
    
    return dependencies;
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

const storeData = (data, json_file_path) => {
    try {
        var obj = Object.fromEntries(data);
        fs.writeFileSync(json_file_path, JSON.stringify(obj, null, 4))
    } catch (err) {
        console.error(err)
    }
}

// var project_list =['lodash']
// var project_list = ['node-mkdirp', 'core', 'tslib', 'classnames', 'async', 'minimist', 'underscore', 'chalk', 'node-semver', 'generator', 'webpack', 'bluebird', 'through2', 'Inquirer.js', 'body-parser', 'colors.js', 'node-glob', 'node-fs-extra', 'commander.js', 'formula-parser_old', 'prop-types', 'request', 'yargs', 'rxjs', 'babel-runtime', 'uuid', 'react', 'package', 'debug']

var project_list = ['lodash', 'underscore', 'eslint-plugin-import', 'mocha', 'angular', 'ramda', 'shelljs', 'js-yaml', 'commander.js', 'bluebird', 'express', 'async', 'jsPDF', 'jquery', 
'coffeescript', 'handlebars.js', 'bootstrap', 'immutable-js', 'package', 'q', 'formula-parser', 'atompm', 'request', 'mongoose', 'mathjs', 'less.js', 'webpack', 
'eslint-plugin-react', 'axios', 'react']
var project_list = ['winston', 'ws', 'qs', 'node-fs-extra', 'UglifyJS']
// var project_list = ['chai', 'autoprefixer', 'winston', 'node-semver', 'html-webpack-plugin', 'qs', 'chokidar', 'postcss', 'postcss-loader', 'ejs', 'morgan', 'url-loader']
// var project_list = ['mysql', 'joi','node-jsonwebtoken', 'create-react-app', 'react-router', 'UglifyJS']
try {
    let filepath = "../libraries";
    var files = fs.readdirSync(filepath);
    
    for(let file_ of files){
        // l(file_)
        if(project_list.includes(file_)){
            json_file = 'connected_files/'+file_+"_dependency_graph.json"
            if(!fs.existsSync(json_file)){
                l(file_)
                let file_path = "../libraries/"+file_;
                main_lst=[]
                global_id = 1;
                global_dict = Object.create(null);
                dict_for_ident = Object.create(null);
                get_file(file_path, main_lst);
                console.log("Get File Done ====>")
                const graph = new Map();
                for(let file_path of main_lst){
                    if (!graph.has(file_path)) {
                        if(!file_path.includes('node_modules') && !file_path.includes('babel_instrumentor.js')){
                            try {
                                var data = fs.readFileSync(file_path).toString();
                                var ast = esprima.parseModule(data, { comment: false, loc: true  });
                                const dependencies = produce_dataset(ast, file_path);
                                graph.set(file_path, dependencies);
                            } catch (e) {
                            }
                        }
                    }
                }
                storeData(graph, 'connected_files/'+file_+"_dependency_graph.json")
            }
        }
    }
    
} catch (e) {
    throw Error(e);
}

