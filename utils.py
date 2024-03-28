import sys
import os
import torch
import random
from scipy import sparse
import dgl
import numpy as np
import pandas as pd
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
import scipy.sparse as sp
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx
from snfpy import snf
import argparse
import scipy.io as sio
import time

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()


def GIP_kernel(Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(Asso_RNA_Dis):
    # calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r



def load_cda_data():
    seq_sim = pd.read_csv("circrna_drug_data/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    str_sim= pd.read_csv("circrna_drug_data/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    A=pd.read_csv("circrna_drug_data/association.csv", index_col=0, dtype=np.float32).to_numpy()
    #A = np.loadtxt("circrna_drug_data/interaction.txt", dtype=int, delimiter=" ")
    return A,seq_sim,str_sim


def get_sfn_sim(A, seq_sim, str_sim):
    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)
    circrna_features, drug_features = [], []
    circrna_features.append(seq_sim)
    circrna_features.append(GIP_c_sim)
    drug_features.append(str_sim)
    drug_features.append(GIP_d_sim)

    circ_fused_network = snf.snf(circrna_features, K=20)
    drug_fused_network = snf.snf(drug_features, K=20)
    return circ_fused_network,drug_fused_network


def construct_network_by_threshold(A,c_sim,d_sim,c_threshold,d_threshold):
    c_network = sim_thresholding(c_sim, c_threshold)
    d_network = sim_thresholding(d_sim, d_threshold)
    network = np.vstack((np.hstack((c_network, A)), np.hstack((A.transpose(), d_network))))
    return c_network,d_network,network

def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    return matrix_copy

def preprocess_graph(adj):
    #adj[np.eye(len(adj),dtype=np.bool_)]=0
    coo_adj = sp.coo_matrix(adj)
    coo_adj.eliminate_zeros()
    edge_index=[coo_adj.row, coo_adj.col]
    return adj, edge_index


def del_association(A, c_node_num, d_node_num, to_del_row, to_del_col):
    new_A=A.copy()
    for index in range(0,len(to_del_row)):
        row=to_del_row[index]
        col=to_del_col[index]
        if row<c_node_num and col>(c_node_num-1) and col<(c_node_num+d_node_num):
            d_row=row
            d_col = col - c_node_num
            if new_A[d_row,d_col]==1 :
                new_A[d_row, d_col] = 0
        if row<c_node_num and col>(c_node_num-1) and col<(c_node_num+d_node_num):
            d_row=row
            d_col = col - c_node_num
            if new_A[d_row,d_col]==1 :
                new_A[d_row, d_col] = 0
    return new_A


def select_asso_predict(edge_label_index,test_y,test_pred,c_node_num, d_node_num):
    pos_cnt=0
    neg_cnt=0
    predict_matrix1 = np.zeros((c_node_num, d_node_num))
    y_matrix1=np.zeros((c_node_num,d_node_num))
    predict_matrix2 = np.zeros((d_node_num, c_node_num))
    y_matrix2 = np.zeros((d_node_num, c_node_num))
    axis=[]
    save_y=[]
    save_pred=[]
    rows=edge_label_index[0]
    cols=edge_label_index[1]
    for index in range(0,int(len(test_pred)/2)):
        row = rows[index]
        col = cols[index]
        if row<c_node_num  and col > (c_node_num - 1) and col < (c_node_num + d_node_num):
            predict_matrix1[row, col - c_node_num] = test_pred[index]
            y_matrix1[row, col - c_node_num] = test_y[index]
            axis.append([row, col - c_node_num])
            pos_cnt+=1
        elif col<c_node_num  and row > (c_node_num - 1) and row < (c_node_num + d_node_num):
            predict_matrix2[row - c_node_num, col] = test_pred[index]
            y_matrix2[row - c_node_num, col] = test_y[index]
            axis.append([col,row-c_node_num])
            pos_cnt+=1
    for index in range(int(len(test_y)/2),len(test_y)):
        row = rows[index]
        col = cols[index]
        if neg_cnt==pos_cnt:
            break
        elif row<c_node_num  and col > (c_node_num - 1) and col < (c_node_num + d_node_num):
            predict_matrix1[row, col - c_node_num] = test_pred[index]
            y_matrix1[row, col - c_node_num] = test_y[index]
            axis.append([row, col - c_node_num])
            neg_cnt+=1
        elif col<c_node_num  and row > (c_node_num - 1) and row < (c_node_num + d_node_num):
            predict_matrix2[row - c_node_num, col] = test_pred[index]
            y_matrix2[row - c_node_num, col] = test_y[index]
            axis.append([col,row-c_node_num])
            neg_cnt+=1
    predict_matrix2=predict_matrix2.transpose()
    y_matrix2=y_matrix2.transpose()
    for i in range(0,271):
        for j in range(0,218):
            if predict_matrix2[i,j]>0:
                if predict_matrix1[i,j]>0:
                    predict_matrix1[i,j]=(predict_matrix1[i,j]+predict_matrix2[i,j])/2
                else:
                    predict_matrix1[i,j]=predict_matrix2[i,j]
    for i in range(0,271):
        for j in range(0,218):
            if y_matrix2[i,j]==1:
                if y_matrix1[i,j]==0:
                    y_matrix1[i,j]=1
    for l in range(0,len(axis)):
        save_y.append(y_matrix1[axis[l][0],axis[l][1]])
        save_pred.append(predict_matrix1[axis[l][0], axis[l][1]])
    return save_y,save_pred,predict_matrix1,y_matrix1,axis


def get_test_result(edge_label_index, snf_test_y, snf_test_pred,graph_test_y, graph_test_pred,c_node_num, d_node_num):
    snf_y,snf_pred,snf_pred_matrix, snf_y_matrix,snf_axis =select_asso_predict(edge_label_index,snf_test_y,snf_test_pred,c_node_num,d_node_num)
    graph_y, graph_pred, graph_pred_matrix, graph_y_matrix, graph_axis = select_asso_predict(edge_label_index, graph_test_y, graph_test_pred, c_node_num, d_node_num)

    snf_y = np.array(snf_y)
    snf_pred = np.array(snf_pred)
    graph_y=np.array(graph_y)
    graph_pred = np.array(graph_pred)
    snf_pred_matrix=np.array(snf_pred_matrix)
    snf_y_matrix=np.array(snf_y_matrix)
    graph_y_matrix=np.array(graph_y_matrix)
    graph_pred_matrix=np.array(graph_pred_matrix)
    snf_axis=np.array(snf_axis)
    graph_axis=np.array(graph_axis)

    save_y = []
    save_pred = []

    y_matrix=snf_y_matrix
    predict_matrix=(snf_pred_matrix+graph_pred_matrix)/2

    for l in range(0, len(snf_axis)):
        save_y.append(y_matrix[snf_axis[l][0], snf_axis[l][1]])
        save_pred.append(predict_matrix[snf_axis[l][0], snf_axis[l][1]])

    return np.array(save_y),np.array(save_pred)

def reset_test_neg_edge_index(test_data,A,c_node_num):
    neg_test_data = test_data.neg_edge_label_index.cpu().numpy().transpose()
    neg_test_data = reconstruct_neg_edge(list(np.argwhere(A == 0)), c_node_num, len(neg_test_data))
    test_data.neg_edge_label_index = torch.tensor(np.array(neg_test_data).transpose(), dtype=torch.long)
    return test_data


def reconstruct_neg_edge(data0_index,c_node_num,sample_num):
    samples = random.sample(data0_index, sample_num)
    rec_samples=[]
    for index in range(0,len(samples)):
        rec_samples.append([samples[index][0], samples[index][1]+c_node_num])
    return rec_samples


def reconstruct_feature_mat(A, seq_sim, str_sim, c_node_num, d_node_num, test_data,cth,dth):
    new_A = del_association(A, c_node_num, d_node_num, test_data.pos_edge_label_index[0],
                            test_data.pos_edge_label_index[1])
    new_c_sim, new_d_sim = get_sfn_sim(new_A, seq_sim, str_sim)
    homo_graph,hetero_graph =load_homo(new_A,new_c_sim,new_d_sim,cth,dth)
    return homo_graph,hetero_graph,new_A,np.hstack((new_c_sim,new_A)),np.hstack((new_d_sim,new_A.transpose()))

def load_homo(assocition,c_sim,d_sim,c_threshold,d_threshold):
    circrna_drug= assocition
    drug_circrna = circrna_drug.T
    drug_drug = sim_thresholding(d_sim, d_threshold)
    circrna_circrna = sim_thresholding(c_sim, c_threshold)

    num_nodes_dict = {'drug': 218, 'circrna': 271}
    dg_data_dict = {
        ('drug', 'similarity', 'drug'): (
        torch.tensor(np.nonzero(drug_drug)[0]), torch.tensor(np.nonzero(drug_drug)[1])),
        ('drug', 'dc', 'circrna'): (
        torch.tensor(np.nonzero(drug_circrna)[0]), torch.tensor(np.nonzero(drug_circrna)[1]))
    }
    cg_data_dict = {
        ('circrna', 'similarity', 'circrna'): (
        torch.tensor(np.nonzero(circrna_circrna)[0]), torch.tensor(np.nonzero(circrna_circrna)[1])),
        ('circrna', 'cd', 'drug'): (
        torch.tensor(np.nonzero(circrna_drug)[0]), torch.tensor(np.nonzero(circrna_drug)[1]))
    }
    dg = dgl.heterograph(dg_data_dict,num_nodes_dict)
    cg = dgl.heterograph(cg_data_dict,num_nodes_dict)
    ds = dgl.from_scipy(sparse.csr_matrix(drug_drug))
    cs = dgl.from_scipy(sparse.csr_matrix(circrna_circrna))

    return [ds,cs],[dg, cg]





