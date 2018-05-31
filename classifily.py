import Crob_Img as cb
import tensorflow as tf
import scipy.io
imagePath = 'gene1.jpg'
#def weight_variable(shape, name1):
#  initial = tf.truncated_normal(shape, stddev=0.1)
#  return tf.Variable(initial, name=name1)
#
#def bias_variable(shape):
#  initial = tf.constant(0.1, shape=shape)
#  return tf.Variable(initial)
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
def save_feature():# name is path+name file ex. "D:/feature"
    graph = load_graph("D:\\work\\face-rec-api\\save_train\\my_model.pb")
    data_mat= scipy.io.loadmat('data/'+str(1)+'.mat')
   # im = np.reshape(np.array(Image.open("crop_gene.png")),(1,224,224,3))
    for op in graph.get_operations():
        print(op.name)
    x = graph.get_tensor_by_name('prefix/Placeholder_18:0')
    y = graph.get_tensor_by_name('prefix/add_8:0')
#      
##     We launch a Session
    with tf.Session(graph=graph) as sess:
#         # Note: we don't nee to initialize/restore anything
#         # There is no Variables in this graph, only hardcoded constants 
#        
         y_out = sess.run(y, feed_dict={
         x: data_mat['y'][0,:]
         })
         # I taught a neural net to recognise when a sum of numbers is bigger than 45
         # it should return False in this case
#         scipy.io.savemat(name+".mat",dict(y_out=y_out))
save_feature()