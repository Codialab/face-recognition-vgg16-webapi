import os
from flask import Flask, render_template, request
from database import *
import Crob_Img as cb
import tensorflow as tf
import numpy as np
import scipy.io
import train as tr
UPLOAD_FOLDER_CLASSIFY = 'data/data_classify'
UPLOAD_FOLDER = 'tmp'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_feature(im):# name is path+name file ex. "D:/feature"
    graph = load_graph("D:\\work\\face-rec-api\\Freeze\\feature.pb")
    im = cb.Crop(im)
    
    if (im == None):
        return None
   # im = np.reshape(np.array(Image.open("crop_gene.png")),(1,224,224,3))
    x = graph.get_tensor_by_name('prefix/permute_1_input:0')
    y = graph.get_tensor_by_name('prefix/flatten_1/Reshape:0')
      
#     We launch a Session
    with tf.Session(graph=graph) as sess:
         # Note: we don't nee to initialize/restore anything
         # There is no Variables in this graph, only hardcoded constants 
        
         y_out = sess.run(y, feed_dict={
         x: im
         })   
         # I taught a neural net to recognise when a sum of numbers is bigger than 45
         # it should return False in this case
#         scipy.io.savemat(name+".mat",dict(y_out=y_out))
    return y_out

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_CLASSIFY'] = UPLOAD_FOLDER_CLASSIFY

@app.route('/train-network', methods = ['GET', 'POST'])
def train_network():
    if request.method == 'POST':
#        f = request.files['file_classify']
        index_max=database_max()
        tr.train(index_max,50)
#        f.save(os.path.join(app.config['UPLOAD_FOLDER_CLASSIFY'], f.filename))
        return 'train succuessful'
    
@app.route('/train')
def train_function():
   return render_template('train.html')

@app.route('/upload-file-classify', methods = ['GET', 'POST'])
def upload_file_classify():
    if request.method == 'POST':
        f = request.files['file_classify']
        f.save(os.path.join(app.config['UPLOAD_FOLDER_CLASSIFY'], f.filename))
        feature = get_feature(os.path.join('data/data_classify', f.filename))
        
        
        if (feature == None):
            os.remove(os.path.join('data/data_classify', f.filename))
            return 'face not found'

        prob = np.squeeze(tr.classifily(feature))
        classify_index = np.argmax(prob)
        name = database_get_name(classify_index)
        print (prob)
        return name + ' , has prob = :' + str(prob[classify_index]*100)+'%' 
    
@app.route('/upload-file-update', methods = ['GET', 'POST'])
def upload_file_update():
    if request.method == 'POST':
        dataset = request.files.getlist('dataset')
        class_name = request.form['class_name']
        table_index = database(class_name)
        os.makedirs('data/'+ str(table_index)) #create folder at directory path
        for file in dataset:
            file.save(os.path.join('data/'+ str(table_index), file.filename))
            y=get_feature(os.path.join('data/'+ str(table_index), file.filename))
            os.remove(os.path.join('data/'+ str(table_index), file.filename))
                            
            if (y == None):
                return 'face not found'
            else:
                scipy.io.savemat('data/'+ str(table_index)+".mat",dict(y=y))
            
            return 'Upload Successful'
    
@app.route('/add-class', methods = ['GET', 'POST'])
def add_class():
    if request.method == 'POST':
        return render_template('upload_file_update.html')
    
@app.route('/addpic-class', methods = ['GET', 'POST'])    
def addpic_class():
    if request.method == 'POST':
        return render_template('upload_file_at_class.html')

@app.route('/upload-file-at-class', methods = ['GET', 'POST'])
def upload_file_at_class():
    if request.method == 'POST':
        dataset = request.files.getlist('dataset') #image from web API
        class_name = request.form['class_name'] #class name from web API
        table_index = database_check(class_name)
        
        if (table_index != -1): #class not define
            for file in dataset:
                file.save(os.path.join('data/'+ str(table_index), file.filename)) #save file
                y=get_feature(os.path.join('data/'+ str(table_index), file.filename))
                os.remove(os.path.join('data/'+ str(table_index), file.filename)) #delete image
                
                if (y == None):
                    return 'face not found'
            
                data = scipy.io.loadmat('data/'+str(table_index)+'.mat') #save file in .mat file
                data['y']=np.append(data['y'],y, axis=0)
                scipy.io.savemat('data/'+ str(table_index)+".mat",data)
                
                return 'Upload Successful'
        else :
            return 'Class not define'
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug = False)