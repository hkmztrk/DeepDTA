from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn

### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2
import pandas as pd
from testdatahelper import *



TABSY = "\t"
figdir = "figures/"

def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))


    encode_smiles= Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles) #pool_size=pool_length[i]


    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    predictions = Dense(1, kernel_initializer='normal')(FC2) 

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_onehot.png')

    return interactionModel





def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII 
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel



def build_single_drug(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XTmodel = Sequential()
    XTmodel.add(Activation('linear', input_shape=(FLAGS.target_count,)))


    encode_smiles = Sequential()
    encode_smiles.add(Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)) 
    encode_smiles.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SMI_LEN, SMI_EMBEDDING_DIMS)
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    encode_smiles.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([encode_smiles, XTmodel], mode='concat', concat_axis=1))
    #interactionModel.add(layers.merge.Concatenate([XDmodel, XTmodel]))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu')) #1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu')) 


    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_drug.png')

    return interactionModel


def build_single_prot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    interactionModel = Sequential()
    XDmodel = Sequential()
    XDmodel.add(Activation('linear', input_shape=(FLAGS.drugcount,)))


    XTmodel1 = Sequential()
    XTmodel1.add(Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128,  input_length=FLAGS.max_seq_len))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)) #input_shape=(MAX_SEQ_LEN, SEQ_EMBEDDING_DIMS)
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1))
    XTmodel1.add(GlobalMaxPooling1D())


    interactionModel.add(Merge([XDmodel, XTmodel1], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_protein.png')

    return interactionModel

def build_baseline(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()

    XDmodel = Sequential()
    XDmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.drug_count, )))

    XTmodel = Sequential()
    XTmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.target_count,)))


    interactionModel.add(Merge([XDmodel, XTmodel], mode='concat', concat_axis=1))

    # Fully connected 
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_baseline.png')

    return interactionModel

def nfold_1_2_3_setting_sample(tr_XD, tr_XT,  tr_Y, te_XD, te_XT, te_Y,  measure, runmethod,  FLAGS, dataset):

    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    
    ### MODIFIED FOR SINGLE TRAIN AND TEST #####
    train_set = outer_train_sets
    #train_set = [item for sublist in outer_train_sets for item in sublist]

    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(tr_XD, tr_XT,  tr_Y, te_XD, te_XT, te_Y,  
                                                                                                measure, runmethod, FLAGS, train_set, test_set)
    
    testperf = all_predictions[bestparamind]##pointer pos 

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.


    foldperf = all_predictions[bestparamind]
    foldloss = all_losses[bestparamind]
    testperfs.append(foldperf)
    testloss.append(foldloss)
    avgperf += foldperf

    avgperf = avgperf / 1
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd




def general_nfold_cv(tr_XD, tr_XT,  tr_Y, te_XD, te_XT, te_Y,  prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                 #100
    batchsz = FLAGS.batch_size                             #256

    logging("---Parameter Search-----", FLAGS)


    ### MODIFIED FOR SINGLE TRAIN



    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [0 for y in range(h)] 
    all_losses = [0 for y in range(h)] 


    valinds = val_sets
    labeledinds = labeled_sets

    tr_label_row_inds, tr_label_col_inds = np.where(np.isnan(tr_Y)==False)  #basically finds the point address of affinity [x,y]
    te_label_row_inds, te_label_col_inds = np.where(np.isnan(te_Y)==False)  #basically finds the point address of affinity [x,y]

    Y_train = np.mat(np.copy(tr_Y))

    params = {}
    XD_train = tr_XD
    XT_train = tr_XT
    trrows = tr_label_row_inds[labeledinds]
    trcols = tr_label_col_inds[labeledinds]

        #print("trrows", str(trrows), str(len(trrows)))
        #print("trcols", str(trcols), str(len(trcols)))

    XD_train = tr_XD[trrows]
    XT_train = tr_XT[trcols]


    train_drugs, train_prots,  train_Y = prepare_interaction_pairs(tr_XD, tr_XT, tr_Y, trrows, trcols)
        
    terows = te_label_row_inds[valinds]
    tecols = te_label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

    val_drugs, val_prots,  val_Y = prepare_interaction_pairs(te_XD, te_XT,  te_Y, terows, tecols)


    pointer = 0
       
    for param1ind in range(len(paramset1)): #hidden neurons
        param1value = paramset1[param1ind]
        for param2ind in range(len(paramset2)): #learning rate
            param2value = paramset2[param2ind]

            for param3ind in range(len(paramset3)):
                param3value = paramset3[param3ind]

                gridmodel = runmethod(FLAGS, param1value, param2value, param3value)

                # Set callback functions to early stop training and save the best model so far
                callbacks = [EarlyStopping(monitor='val_loss', patience=15)]

                gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch,  
                	shuffle=False ) 
                #validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)), 

                predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                json.dump(predicted_labels.tolist(), open("predicted_labels_"+str(pointer)+ ".txt", "w"))
                loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                rperf = prfmeasure(val_Y, predicted_labels)
                #rperf = rperf[0]


                logging("P1 = %d,  P2 = %d, P3 = %d,  CI-i = %f, CI-ii = %f, MSE = %f" % 
                (param1ind, param2ind, param3ind,  rperf, rperf2, loss), FLAGS)

                #plotLoss(gridres, param1ind, param2ind, param3ind, "1")

                all_predictions[pointer] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                all_losses[pointer]= loss

                pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.

                    foldperf = all_predictions[pointer]
                    avgperf += foldperf
                    #avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1
        
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses



def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


   
def plotLoss(history, batchind, epochind, param3ind, foldind):

    figname = "b"+str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_"  + str( foldind) + "_" + str(time.time()) 
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)



def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):

            drug = XD[rows[pair_ind]]
            drugs.append(drug)

            target=XT[cols[pair_ind]]
            targets.append(target)

            affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity


       
def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.train_path,
    				   fpath_test = FLAGS.test_path,
                      setting_no = FLAGS.problem_type, 
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 

    #XD, XT, Y = dataset.parse_data(fpath = FLAGS.dataset_path)
    tr_XD, tr_XT, tr_Y, te_XD, te_XT, te_Y = dataset.parse_train_test_data(FLAGS)

    tr_XD = np.asarray(tr_XD)
    tr_XT = np.asarray(tr_XT)
    tr_Y = np.asarray(tr_Y)

    te_XD = np.asarray(te_XD)
    te_XT = np.asarray(te_XT)
    te_Y = np.asarray(te_Y)

    tr_drugcount = tr_XD.shape[0]
    print("train drugs: ", tr_drugcount)
    tr_targetcount = tr_XT.shape[0]
    print("train targets: ", tr_targetcount)

    te_drugcount = te_XD.shape[0]
    print("test drugs: ", te_drugcount)
    te_targetcount = te_XT.shape[0]
    print("test targets: ", te_targetcount)

    FLAGS.drug_count = tr_drugcount
    FLAGS.target_count = tr_targetcount



    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(tr_XD, tr_XT,  tr_Y, te_XD, te_XT, te_Y,
                                                                     perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" % 
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)




def run_regression( FLAGS ): 

    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)




if __name__=="__main__":

    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    prepare_new_data(FLAGS.test_path, test=True)
    #prepare_new_data(FLAGS.test_path, test=False) #Uncomment this if you also have a new training data

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )