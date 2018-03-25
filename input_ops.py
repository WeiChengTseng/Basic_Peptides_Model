import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import pickle
import time
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, num_training_data, num_testing_data=32, word_dim=10):
        """
        Initialize the dataset.
        
        Inputs:
        - num_training_data: the number of training data.
        - num_testing_data: the number of testing data.
        - word_dim: the word dimension of word2vec.
        """
        self.num_labels, self.num_peptides, self.num_testing_data = None, None, num_testing_data
        self.BP, self.CC, self.MF = {}, {}, {}
        self.GO_terms, self.peptides, self.word2vec_table = {}, {}, {}
        self.word_dim = word_dim

        s = time.time()
        self.load_data()
        e = time.time()
        print('It takes', e-s, 'seconds to load data.')
        return

    def load_data(self):
        """
        Load all the data.
        """

        self.peptides = pickle.load(open('data/peptides_re_all.p', 'rb'))
        self.GO_terms = pickle.load(open('data/GO_terms_re_all.p', 'rb'))
        self.word2vec_table = pickle.load(open('data/word2vec_table.p', 'rb'))
        self.X_index = pickle.load(open('data/X_index_re.p', 'rb'))
        self.Y_index = pickle.load(open('data/Y_index_all.p', 'rb'))
        self.BP = pickle.load(open('data/GO_BP_re.p', 'rb'))
        self.CC = pickle.load(open('data/GO_CC_re.p', 'rb'))
        self.MF = pickle.load(open('data/GO_MF_re.p', 'rb'))
        self.num_labels, self.num_peptides = len(self.GO_terms), len(self.peptides)
        self.peptides, self.GO_terms = None, None
        
        for i in self.word2vec_table:
            self.word2vec_table[i] /= 100
        
        print('All data are loaded!')
        return

    def get_training_data(self, nth_batch, batch_size=32, max_len=608):
        """
        Get training data and testing data.
        
        Input:
        - nth_batch: the n-th batch of the training data.
        - batch_size: batch size of the training data.
        - max_len: the max length of input data sequence.

        Output:
        - x_train: training peptides
        - y_train: training labels
        """

        x_train, y_train = [], np.zeros((batch_size, self.num_labels), dtype=np.int8)
        zeros = [0] * self.word_dim
        for i in self.X_index[nth_batch*batch_size: (nth_batch+1) * batch_size]:
            peptide = []
            for count, j in enumerate(i):
                peptide.append(list(self.word2vec_table[j]))
                if count >= max_len-1:
                    break
            if count < max_len-1:
                for i in range(max_len-count-1):
                    peptide.append(zeros)
            x_train.append((peptide))
        for i in range(batch_size):
            y_train[i, self.Y_index[nth_batch*batch_size+i]] = 1

        return np.array(x_train), y_train

    def get_testing_data(self, max_len=608):
        """
        Get training data and testing data.

        Input:
        - max_len: the max length of input data sequence.
        
        Output:
        - x_test: testing peptides
        - y_test: testing labels
        """

        x_test, y_test = [], np.zeros((self.num_testing_data, self.num_labels), dtype=np.int8)
        zeros = [0] * self.word_dim
        for i in self.X_index[-self.num_testing_data: ]:
            peptide = []
            for count, j in enumerate(i):
                peptide.append(list(self.word2vec_table[j]))
                if count >= max_len-1:
                    break
            if count < max_len-1:
                for i in range(max_len-count-1):
                    peptide.append(zeros)
            x_test.append(np.array(peptide))

        for i in range(self.num_testing_data):
            y_test[i, self.Y_index[-self.num_testing_data+i]] = 1

        return np.array(x_test), y_test

    def GO_term_info(self, GO_id, visualization = False):
        """
        Show the information of the GO term.

        Input:
        - GO_id: the GO term we want to know
        - visualization: whether to visual the GO term
        """

        print('ID:', GO_id)
        print('name:', self.GO_terms[GO_id]['name'])
        print('namespace:', self.GO_terms[GO_id]['namespace'])
        print('is_a:', self.GO_terms[GO_id]['is_a'])
        print('children:', self.GO_terms[GO_id]['children'])
        if visualization:
            self.visualize_label(GO_id)

        return

    def visualize_label(self, GO_id):
        """
        Visualize the relationship of the GO term by using networkx.

        Input:
        - GO_id: the GO term we want to know
        """

        start, end, added = [], [], []
        ancestors = self.back_to_root(GO_id)
        for i in ancestors:
            for k in self.GO_terms[i]['children']:
                if k in ancestors:
                    start.append(i)
                    end.append(k)

        DAG = nx.DiGraph()
        for i in start:
            if i not in added:
                DAG.add_node(i)
        for i in end:
            if i not in added:
                DAG.add_node(i)
        for i in range(len(start)):
            DAG.add_edge(start[i], end[i])

        pos =graphviz_layout(DAG, prog='dot')
        fig = plt.figure(figsize=(12,12)) 
        nx.draw(DAG, pos, with_labels=True, arrows=True, font_size=7, node_size=2200, 
        node_color="skyblue", label='GO terms', edge_color='white')
        fig.set_facecolor('#1e1e1e')
        plt.show()
        
        return
    
    def back_to_root(self, GO_id):
        """
        Find all the labels from GO_id to the root of the DAG by calling 
        self.back_to_root_step

        Input:
        - GO_id: the GO term we want to start

        Output:
        - passed: a list that store all the labels from GO_id to the root
        """
        passed = []

        if GO_id in self.GO_terms:
            if GO_id in self.BP.keys():
                namespace = 'biological_process'
            elif GO_id in self.CC.keys():
                namespace = 'cellular_component'
            elif GO_id in self.MF.keys():
                namespace = 'molecular_function'
            passed = self.back_to_root_step(GO_id, namespace, start=True)
            return passed
        else:
            passed.append(GO_id)
            return passed

    def back_to_root_step(self, GO_id, namespace, passed=[], start=False):
        """
        Find the ancestors of GO_id recursively step by step.

        Input:
        - GO_id: the GO term we want to start.
        - namespace: the nampspace that GO_id belongs to.
        - passed: a list that store all the labels that already passed currently.
        - start: call the function first time.

        Output: 
        - passed: a list that store all the labels that already passed after this step.
        """
        if start:
            passed = []
        if namespace == 'biological_process':
            if GO_id not in passed:
                passed.append(GO_id)
            if self.BP[GO_id]['is_a'] != []:
                for i in self.BP[GO_id]['is_a']:
                    self.back_to_root_step(i, namespace, passed)
                return passed
            else:
                return passed
        elif namespace == 'cellular_component':
            if GO_id not in passed:
                passed.append(GO_id)
            if self.CC[GO_id]['is_a'] != []:
                for i in self.CC[GO_id]['is_a']:
                    self.back_to_root_step(i, namespace, passed)
                return passed
            else:
                return passed
        elif namespace == 'molecular_function':
            if GO_id not in passed:
                passed.append(GO_id)
            if self.MF[GO_id]['is_a'] != []:
                for i in self.MF[GO_id]['is_a']:
                    self.back_to_root_step(i, namespace, passed)
                return passed
            else:
                return passed
