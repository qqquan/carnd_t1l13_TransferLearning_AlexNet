import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np


# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    dataset_train_raw = pickle.load(f)

X_raw = dataset_train_raw['features']

y_raw = dataset_train_raw['labels']

y_raw_ClassIdLs = list(set(y_raw)) #list of all class id
y_raw_ClassNum = len(y_raw_ClassIdLs)

y_ClassNum = y_raw_ClassNum

y_raw_OneHot = np.eye(y_ClassNum)[y_raw]

print("Input Shape: {}".format(X_raw.shape))

### Data exploration visualization goes here.
if 0:
	import matplotlib.pyplot as plt

	def findRandomTrainingImage(img_dataset):
	    idx = int(img_dataset.shape[0] * np.random.random())
	    image_np = np.array(img_dataset[idx])
	    return image_np
	    

	plt.rcParams['figure.figsize'] = (8, 5) # width, height

	plt.figure(1)
	plt.suptitle('Training Examples')

	plt.subplot(221)
	img = findRandomTrainingImage(X_raw)
	plt.imshow(img)

	plt.subplot(222)
	img = findRandomTrainingImage(X_raw)
	plt.imshow(img)

	plt.subplot(223)
	img = findRandomTrainingImage(X_raw)
	plt.imshow(img)

	plt.subplot(224)
	img = findRandomTrainingImage(X_raw)
	plt.imshow(img)

	plt.show()

# Normalize data
# Based on AlexNet paper: We did not pre-process the images
# in any other way, except for subtracting the mean activity over the training set from each pixel. So
# we trained our network on the (centered) raw RGB values of the pixels.
raw_mean = np.mean(X_raw)
X_raw_norm = (X_raw - raw_mean)


# Split data into training and validation sets.

import sklearn.model_selection

#reduce training size
DBG_TrainingSize = 128
print('Debugging Size: ',DBG_TrainingSize)
X_raw = X_raw[:DBG_TrainingSize]
y_raw_OneHot = y_raw_OneHot[:DBG_TrainingSize]


X_Train, X_Vali, y_Train_OneHot ,y_Vali_OneHot =  \
            sklearn.model_selection.train_test_split(X_raw, 
                                                     y_raw_OneHot, 
                                                     train_size= 0.75, 
                                                     stratify = y_raw_OneHot 
                                                    )


# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y_output = tf.placeholder(tf.float32, [None, y_ClassNum]) 
# Resize the images so they can be fed into AlexNet.
x_resized = tf.image.resize_images(x,(227, 227))


nn_input = x
nn_output = y_output

InputSet_Validation = X_Vali
OutputLabel_Validation = y_Vali_OneHot
# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.

def TF_getFeatureNum(input_tensor):
    
    t_shape = input_tensor.get_shape()    
    if(t_shape.ndims>1):
        feature_num = t_shape[1:].num_elements()
    elif (t_shape.ndims == 0):
        raise Exception('0-D tensor is not useful to train model!') 
    
    return feature_num

fc7_OutputSize = TF_getFeatureNum(fc7)

w = tf.Variable(tf.truncated_normal(shape=[fc7_OutputSize, y_ClassNum],stddev=0.0001))
b = tf.Variable(tf.constant(0.0, shape=[y_ClassNum]))


fc8_layer = tf.add(tf.matmul(fc7, w), b)

logits = tf.nn.relu(fc8_layer)   


# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_output))


    
# cost = tf.reduce_mean(-tf.reduce_sum(y_output * tf.log(tf.clip_by_value(tf.nn.softmax(logits),1e-10,1.0)), reduction_indices=[1]))
# GradientDescentOptimizer() is slower in driving down lost. Based on my test of 20 epochs, Adam acheives >90% after 10 epochs, while regular gradient descent optimizer attains only 6% accuracy.  
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_output, 1))
# Calculate validation accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Train and evaluate the feature extraction model.
# Launch the graph"

import logging
logging.basicConfig(filename='log_test_result.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def findNextDatasetBatch(batch_idx, batch_total, batch_length, ls_input_matrix, ls_output_matrix):
    
    if (batch_idx == (batch_total-1)):
        #last batch does not necessarily have enough examples
        return ls_input_matrix[batch_idx*batch_length:], ls_output_matrix[batch_idx*batch_length:]
    else:
        return ls_input_matrix[batch_idx*batch_length:(batch_idx+1)*batch_length], ls_output_matrix[batch_idx*batch_length:(batch_idx+1)*batch_length]
    
    

def calcDatasetAccuracy(session, batch_size, ls_input_matrix, ls_output_matrix):

    total_num = int(np.ceil(ls_input_matrix.shape[0]/batch_size))
    
    accu_sum_batch = 0
    
    # Loop over all batches
    for i in range(total_num):
        batch_x, batch_y = findNextDatasetBatch(i, total_num, batch_size, ls_input_matrix,  ls_output_matrix)
        
        accu_sum_batch += session.run(accuracy, feed_dict={nn_input: batch_x, nn_output: batch_y})
        
    accu_full = accu_sum_batch/total_num
    
    return accu_full
    
 
def calcFinalAccuracy(session, batch_size, print_enable = False):
    
    
    accu_full_train = calcDatasetAccuracy(session, batch_size, X_Train, y_Train_OneHot )

    logging.info("***Accuracy - Full Training:         {:.2f}%".format( accu_full_train*100) )

    
    
    accu_full_validation = calcDatasetAccuracy(session, batch_size, InputSet_Validation, OutputLabel_Validation )
    
    logging.info("***Accuracy - Full Validation:       {:.2f}%".format( accu_full_validation*100) )
    
    
    

    if(print_enable == True):
        print("Accuracy - Full Training:", accu_full_train)
        print("Accuracy - Full Validation:", accu_full_validation)    

    
def TraffSign_trainModel(sess, epochs, batch_size=64, display_step = 1):
    # Training cycle


    size_train = X_Train.shape[0]

    for epoch in range(epochs):
        total_batch = int(size_train/batch_size) 
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = findNextDatasetBatch(i, total_batch, batch_size, X_Train, y_Train_OneHot )

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={nn_input: batch_x, nn_output: batch_y})
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            
            batch_x, batch_y = findNextDatasetBatch(0, total_batch, batch_size, X_Train, y_Train_OneHot ) # use first batch to test
            
            c = sess.run( cost, feed_dict={nn_input: batch_x, nn_output: batch_y})

            logging.debug("Epoch: {:5d};  cost={:.9f}".format((epoch+1), c))
 
            accu_vali = accuracy.eval(session=sess, feed_dict={nn_input: batch_x, nn_output: batch_y})
    
            logging.debug("Accuracy(training batch): {:.2f}%".format(accu_vali*100)) 
        if epoch % (display_step*10) == 0:
            calcFinalAccuracy(sess, batch_size)
    
    print("Optimization Finished!")
    logging.info("Optimization Finished!")
    # Test model
    
    calcFinalAccuracy(sess, batch_size,  print_enable = True)

    

TraffSign_Session = tf.Session()

TraffSign_Session.run(init)

import time
from datetime import timedelta

time_start = time.time()


TraffSign_trainModel(TraffSign_Session, epochs = 10)


time_end = time.time()
print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
logging.info("Time usage: {}".format(timedelta(seconds=int( time_end - time_start))) )



# Read Images
from scipy.misc import imread

im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)



# Run Inference
t = time.time()

logits_softmax = tf.nn.softmax(logits)

output = TraffSign_Session.run(logits_softmax, feed_dict={x: [im1, im2]})

# Print Output


def TraffSign_loadTrafSignClasNames(file, class_num):

    import csv

    name_list = ['']* class_num
    with open(file) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            k=row['ClassId']
            v= row['SignName']
            name_list[int(k)] = v
    return name_list


class_names = TraffSign_loadTrafSignClasNames('signnames.csv', y_ClassNum)

for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))