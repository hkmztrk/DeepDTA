import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
#from keras.preprocessing.sequence import pad_sequences


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

# CHARPROTSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
#             'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
#             'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
#             'O': 20, 'U': 20,
#             'B': (2, 11),
#             'Z': (3, 13),
#             'J': (7, 9) }
# CHARPROTLEN = 21

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

#  Y = -(np.log10(Y/(math.pow(math.e,9))))

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1 

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()



## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, fpath_test, setting_no, seqlen, smilen, need_shuffle = False):

    ### TODO : ADD SMILES TYPE CHOICE HERE
    self.SEQLEN = seqlen
    self.SMILEN = smilen
    #self.NCLASSES = n_classes
    self.charseqset = CHARPROTSET
    self.charseqset_size = CHARPROTLEN

    self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
    self.charsmiset_size = CHARISOSMILEN
    self.PROBLEMSET = setting_no

    # read raw file

    #self._raw = self.read_sets( fpath, setting_no )

    # iteration flags
    # self._num_data = len(self._raw)


  def read_sets(self, FLAGS): ### fpath should be the dataset folder /kiba/ or /davis/
    print("Reading %s start" % FLAGS.train_path)

    test_fold = json.load(open(FLAGS.test_path + "folds/test_fold.txt"))
    train_folds = json.load(open(FLAGS.train_path + "folds/train_fold.txt"))
    
    return test_fold, train_folds


  def parse_train_test_data(self, FLAGS, with_label=True): 
		
    print("Read %s start" % FLAGS.train_path)

    tr_ligands =  json.load(open(FLAGS.train_path+"ligands_iso.txt"), object_pairs_hook=OrderedDict)
    tr_proteins = json.load(open(FLAGS.train_path+"proteins.txt"), object_pairs_hook=OrderedDict)

    tr_Y = pickle.load(open(FLAGS.train_path + "Y","rb"), encoding='latin1') ### TODO: read from raw
    if FLAGS.isLog:
        tr_Y = -(np.log10(tr_Y/(math.pow(math.e,9))))

    print("Read %s start" % FLAGS.test_path)
    te_ligands =  json.load(open(FLAGS.test_path+"ligands.txt"), object_pairs_hook=OrderedDict)
    te_proteins = json.load(open(FLAGS.test_path+"proteins.txt"), object_pairs_hook=OrderedDict)

    te_Y = pickle.load(open(FLAGS.test_path + "Y","rb"), encoding='latin1') ### TODO: read from raw


    tr_XD = []
    tr_XT = []

    te_XD = []
    te_XT = []

    if with_label:
        for d in tr_ligands.keys():
            tr_XD.append(label_smiles(tr_ligands[d], self.SMILEN, self.charsmiset))

        for t in tr_proteins.keys():
            tr_XT.append(label_sequence(tr_proteins[t], self.SEQLEN, self.charseqset))

        for d in te_ligands.keys():
            te_XD.append(label_smiles(te_ligands[d], self.SMILEN, self.charsmiset))

        for t in te_proteins.keys():
            te_XT.append(label_sequence(te_proteins[t], self.SEQLEN, self.charseqset))
    else:
        for d in tr_ligands.keys():
            tr_XD.append(one_hot_smiles(tr_ligands[d], self.SMILEN, self.charsmiset))

        for t in tr_proteins.keys():
            tr_XT.append(one_hot_sequence(tr_proteins[t], self.SEQLEN, self.charseqset))

        for d in te_ligands.keys():
            te_XD.append(one_hot_smiles(te_ligands[d], self.SMILEN, self.charsmiset))

        for t in te_proteins.keys():
            te_XT.append(one_hot_sequence(te_proteins[t], self.SEQLEN, self.charseqset))



  
    return tr_XD, tr_XT, tr_Y, te_XD, te_XT, te_Y

