import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.models import Sequential, Model
from tensorflow.python.framework import ops
import freeze as fz
import graph as gh
import numpy as np
import scipy.io
import database as db
import os, glob
from random import shuffle
path="D:\\work\\face-rec-api\\save_train\\"
batchsize = 1

#############################################################################################
def is_saved_model_exist(path):
  fileList = os.listdir(path)
  for file in fileList:
    if file.endswith(".meta"):
      if "FaceRec" in file:
        return file
  return ""
#############################################################################################
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
#############################################################################################
def weight_variable(shape, name1):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name1)
#############################################################################################
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#############################################################################################
def network(x,number_Class):
    w = tf.Variable(tf.zeros([2622,number_Class]),name='w1')
    b = tf.Variable(tf.zeros([number_Class]),name='b1')
    y_conv= tf.add(tf.matmul(x,w),b,name ='add_yconv')
    y_out = tf.nn.softmax(y_conv,name='y_prob')
    return y_out
#############################################################################################
def train(number_Class,iteration):
    # delete old save
    for filename in glob.glob(path+"FaceRec*"):
        os.remove(filename) 
    ops.reset_default_graph()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[batchsize,2622],name='Inputx_Placeholder1')
    y_ = tf.placeholder(tf.float32, shape=[batchsize,number_Class],name='Inputy_Placeholder1')
    y_conv = network(x,number_Class)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())       
        #for ...
            #load data trainx,labely = ...      
        for i in range(iteration):
            trainx,labely = load_data()
            acc = 0
            for j in range(np.shape(trainx)[0]): 
                y_label = np.reshape(labely[j,:],(1,number_Class)).astype('float32')
#                print(y_label)
                train_step.run(feed_dict={x: np.reshape(trainx[j,:],(1, 2622)), y_: np.reshape(labely[j,:],(1,number_Class)) })
                train_accuracy = accuracy.eval(feed_dict={x:np.reshape(trainx[j,:],(1, 2622)), y_:y_label })
#                y_out = sess.run(y_conv, feed_dict={ x: np.reshape(trainx[j,:],(1, 2622)) })
#                print(train_accuracy)
                acc = acc + train_accuracy
#                print(y_out)
            acc = acc / np.shape(trainx)[0]
            print("round,"+str(i)+",accuracy,"+str(acc))
            
            
        saver.save(sess, path+"FaceRec")
        load_freeze(number_Class)
        return 1
#############################################################################################3		
def load_data():
    output_label = []
    output_data = []
    index_max=db.database_max() 
    label = []
    data_mat_add= scipy.io.loadmat('data/'+str(1)+'.mat')
    for i in range(np.shape(data_mat_add['y'])[0]):
        l = np.zeros((index_max),dtype=int)
        l[0] = 1
        label.append(l)
    if index_max > 1: 
        for i in range(2,index_max+1):         
            data_mat= scipy.io.loadmat('data/'+str(i)+'.mat')
            data_mat_add['y']=np.append(data_mat_add['y'],data_mat['y'],axis=0)
            for j in range(np.shape(data_mat['y'])[0]):
                l = np.zeros((index_max),dtype=int)
                l[i-1] = 1
                label.append(l)
    #start shuffle
    index_list = list(range(np.shape(data_mat_add['y'])[0]))
    shuffle(index_list)
    for i in index_list:
        output_label.append(label[i])
        output_data.append(data_mat_add['y'][i,:])
    return np.array(output_data),np.array(output_label)
#############################################################################################
def load_freeze(number_Class):
#    w = tf.Variable(tf.zeros([2622,number_Class]),name='w0')
#    b = tf.Variable(tf.zeros([number_Class]),name='b0')
#    y_conv= tf.add(tf.matmul(x,w),b,name ='add_yconv_2_')
    sess = tf.Session()
    filename = is_saved_model_exist(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  
        if filename != "":
            saver = tf.train.import_meta_graph(path+filename)
            saver.restore(sess, tf.train.latest_checkpoint(path))
            frozen_graph = freeze_session(sess,output_names=['y_prob'])
            tf.train.write_graph(frozen_graph, path, "my_model.pb", as_text=False)
            return 1
        else:
            return 0
###############################################################################################
def classifily(intput_feature):# name is path+name file ex. "D:/feature"
    graph=gh.load_graph('D:\\work\\face-rec-api\\save_train\\my_model.pb')
#    im = cb.Crop(im)
#    intput_feature = scipy.io.loadmat('data/'+str(2)+'.mat')
    x = graph.get_tensor_by_name('prefix/Inputx_Placeholder1:0')
    y = graph.get_tensor_by_name('prefix/y_prob:0')

#     We launch a Session
    with tf.Session(graph=graph) as sess:
         # Note: we don't nee to initialize/restore anything
         # There is no Variables in this graph, only hardcoded constants 
         
#         y_out = sess.run(y, feed_dict={
#                 x: intput_feature['y'][0].reshape((1,2622))
#         })
        y_out = sess.run(y, feed_dict={x: intput_feature})
#          I taught a neural net to recognise when a sum of numbers is bigger than 45
#          it should return False in this case
#         scipy.io.savemat(name+".mat",dict(y_out=y_out))
    return y_out
#############################################################################################3

#prob = classifily(input_feature)
#for n in tf.get_default_graph().as_graph_def().node :
#    print(n.name) 
