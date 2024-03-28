import warnings

import numpy as np

...
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm.auto import tqdm
from numpy import interp
# custom modules
from utils import *
from maskgae.model import DPMGCDA, EdgeDecoder, GNNEncoder, FeatureExtracter
from maskgae.mask import MaskEdge,MaskPath
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,precision_recall_curve,auc,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def create_model(maskgae_args,ifGraph=False):
    if maskgae_args.mask == 'Path':
        mask = MaskPath(p=maskgae_args.p, num_nodes=maskgae_args.c_node_num + maskgae_args.d_node_num,
                        start=maskgae_args.start,
                        walk_length=maskgae_args.encoder_layers + 1)
    elif maskgae_args.mask == 'Edge':
        mask = MaskEdge(p=maskgae_args.p)
    else:
        mask = None  # vanilla GAE

    edge_decoder = EdgeDecoder(maskgae_args.hidden_channels, maskgae_args.decoder_channels,
                                   num_layers=maskgae_args.decoder_layers, dropout=maskgae_args.decoder_dropout)
    if ifGraph:
        feature_extracter = FeatureExtracter(maskgae_args.in_size, maskgae_args.feature_size)
        encoder = GNNEncoder(maskgae_args.feature_size[0], maskgae_args.encoder_channels, maskgae_args.hidden_channels,
                             num_layers=maskgae_args.encoder_layers, dropout=maskgae_args.encoder_dropout,
                             bn=maskgae_args.bn, layer=maskgae_args.layer, activation=maskgae_args.encoder_activation)
    else:
        feature_extracter=None
        encoder = GNNEncoder(maskgae_args.in_size[0], maskgae_args.encoder_channels, maskgae_args.hidden_channels,
                             num_layers=maskgae_args.encoder_layers, dropout=maskgae_args.encoder_dropout,
                             bn=maskgae_args.bn, layer=maskgae_args.layer, activation=maskgae_args.encoder_activation)

    model = DPMGCDA(encoder, edge_decoder, mask, feature_extracter).to(maskgae_args.device)

    return model


def association_linkpred(maskgae_args, splits, c_snf_feature,d_snf_feature,homo_graph,c_graph_feature, d_graph_feature,
                     device="cpu"):
    snf_feature_model=create_model(maskgae_args,False)
    graph_feature_model=create_model(maskgae_args,True)

    snf_optimizer = torch.optim.Adam(snf_feature_model.parameters(),
                                 lr=maskgae_args.lr,
                                 weight_decay=maskgae_args.weight_decay)
    graph_optimizer = torch.optim.Adam(graph_feature_model.parameters(),
                                     lr=maskgae_args.lr,
                                     weight_decay=maskgae_args.weight_decay)

    best_valid = 0
    batch_size = maskgae_args.batch_size
    train_data = splits['train'].to(device)
    test_data = splits['test'].to(device)

    node_snf_features = [d_snf_feature.to(device), c_snf_feature.to(device)]

    node_graph_features = [d_graph_feature.to(device), c_graph_feature.to(device)]
    homo_graph = [homo_graph[0].to(device), homo_graph[1].to(device)]
    print('Homogeneous Graph Level Perspective Training!')
    for epoch in tqdm(range(1, 1 + maskgae_args.epochs)):
        loss = graph_feature_model.train_step(train_data,graph_optimizer, homo_graph,node_graph_features,
                                batch_size=maskgae_args.batch_size)
    print('Combined Feature Level Perspective Training!')
    for epoch in tqdm(range(1, 1 + maskgae_args.epochs)):
        loss = snf_feature_model.train_step(train_data, snf_optimizer, homo_graph, node_snf_features,
                                batch_size=maskgae_args.batch_size)
    #model.load_state_dict(torch.load(args.save_path))
    graph_test_y, graph_test_pred = graph_feature_model.test_step(test_data,
                                                                  test_data.pos_edge_label_index,
                                                                  test_data.neg_edge_label_index,
                                                                  homo_graph, node_graph_features,
                                                                  batch_size=batch_size)
    snf_test_y, snf_test_pred = snf_feature_model.test_step(test_data,
                                        test_data.pos_edge_label_index,
                                        test_data.neg_edge_label_index,
                                        homo_graph, node_snf_features,
                                        batch_size=batch_size)
    edge_label_index = np.hstack((test_data.pos_edge_label_index.cpu().numpy(), test_data.neg_edge_label_index.cpu().numpy()))
    y, pred = get_test_result(edge_label_index, snf_test_y,snf_test_pred,graph_test_y,graph_test_pred,
                                                        maskgae_args.c_node_num, maskgae_args.d_node_num)

    return y,pred

def DPMGCDA_train(maskgae_args, device):
    A, seq_sim, str_sim = load_cda_data()
    c_node_num = A.shape[0]
    d_node_num = A.shape[1]
    cth=0.0040
    dth=0.0048
    c_sim,d_sim=get_sfn_sim(A,seq_sim,str_sim)

    c_network,d_network,network = construct_network_by_threshold(A, c_sim, d_sim,cth,dth)
    network, edge_index = preprocess_graph(network)
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
    data = Data(edge_index=edge_index)
    data.x = data.train_mask = data.val_mask = data.test_mask = y = None

    acc_list = []
    prec_list = []
    recall_list = []
    spec_list = []
    f1_score_list = []
    auc_list = []
    aupr_list = []

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    precisions = []
    mean_recall = np.linspace(0, 1, 100)

    for run in range(1, maskgae_args.runs + 1):
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.0, num_test=0.1,
                                                            is_undirected=True,
                                                            split_labels=True,
                                                            add_negative_train_samples=False)(data)

        test_data=reset_test_neg_edge_index(test_data,A,c_node_num)
        homo_graph,hetero_graph,new_A,c_snf_feature,d_snf_feature=reconstruct_feature_mat(A, seq_sim, str_sim, c_node_num, d_node_num, test_data, cth, dth)

        d_snf_feature = torch.tensor(np.array(d_snf_feature),dtype=torch.float32)
        c_snf_feature = torch.tensor(np.array(c_snf_feature),dtype=torch.float32)

        d_graph_feature = torch.randn((d_node_num, maskgae_args.in_size[0]))
        c_graph_feature = torch.randn((c_node_num, maskgae_args.in_size[1]))

        splits = dict(train=train_data, test=test_data)

        y,pred=association_linkpred(maskgae_args, splits,
                                    c_snf_feature,d_snf_feature,
                                    homo_graph,c_graph_feature,d_graph_feature,device=device)

        test_auc=roc_auc_score(y, pred)
        auc_list.append(test_auc)
        precision, recall, _ = precision_recall_curve(y, pred)
        rev_precision = precision[::-1]
        rev_recall = recall[::-1]
        precisions.append(interp(mean_recall, rev_recall, rev_precision))
        precisions[-1][0] = 1.0
        tn,fp,fn,tp=confusion_matrix(y,pred.round()).ravel()
        test_spec=tn/(tn+fp)
        spec_list.append(test_spec)
        test_aupr = auc(recall, precision)
        aupr_list.append(test_aupr)
        test_pre = precision_score(y,pred.round())
        test_rec = recall_score(y,pred.round())
        test_f1_score = f1_score(y,pred.round())
        test_acc = accuracy_score(y,pred.round())
        acc_list.append(test_acc)
        prec_list.append(test_pre)
        recall_list.append(test_rec)
        f1_score_list.append(test_f1_score)
        test_fpr, test_tpr, _ = roc_curve(y, pred, drop_intermediate=False)
        tprs.append(interp(mean_fpr, test_fpr, test_tpr))
        tprs[-1][0] = 0.0


        print(f' Runs {run} : - AUC: {test_auc:.2%}', f'AUPR: {test_aupr:.2%}', f'F1: {test_f1_score:.2%}',
                  f'ACC: {test_acc:.2%}', f'PRE: {test_pre:.2%}', f'REC: {test_rec:.2%}', f'SPE: {test_spec:.2%}', )


    print(f'Link Prediction Results ({maskgae_args.runs} runs):\n'
          f'AUC: {np.mean(auc_list):.2%} ± {np.std(auc_list):.2%}',
          f'AUPR: {np.mean(aupr_list):.2%} ± {np.std(aupr_list):.2%}',
          f'F1_score: {np.mean(f1_score_list):.2%} ± {np.std(f1_score_list):.2%}',
          f'ACC: {np.mean(acc_list):.2%} ± {np.std(acc_list):.2%}',
          f'PRE: {np.mean(prec_list):.2%} ± {np.std(prec_list):.2%}',
          f'REC: {np.mean(recall_list):.2%} ± {np.std(recall_list):.2%}',
          f'SPE: {np.mean(spec_list):.2%} ± {np.std(spec_list):.2%}',
          )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[1] = 1.0


def parse_args():
    maskgaeparser = argparse.ArgumentParser()
    maskgaeparser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, 'Path',`Edge` or `None` (default: Path)")
    maskgaeparser.add_argument('--seed', type=int, default=2024, help='Random seed for model and dataset. (default: 2024)')
    maskgaeparser.add_argument('--c_node_num', type=int, default=271, help='Number of circrna. (default: 271)')
    maskgaeparser.add_argument('--d_node_num', type=int, default=218, help='Numver of drug. (default: 218)')

    maskgaeparser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    maskgaeparser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
    maskgaeparser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
    maskgaeparser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
    maskgaeparser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
    maskgaeparser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
    maskgaeparser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
    maskgaeparser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    maskgaeparser.add_argument('--encoder_dropout', type=float, default=0.2, help='Dropout probability of encoder. (default: 0.2)')
    maskgaeparser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

    maskgaeparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training. (default: 1e-3)')
    maskgaeparser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
    maskgaeparser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    maskgaeparser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

    maskgaeparser.add_argument('--in_size', type=int, default=[489, 489],
                               help='Number of input feature size. (default: 256)')
    maskgaeparser.add_argument('--feature_size', type=int, default=[256,256],
                               help='Number of input feature size. (default: 256)')


    maskgaeparser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
    maskgaeparser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

    maskgaeparser.add_argument('--epochs', type=int, default=600, help='Number of training epochs. (default: 600)')
    maskgaeparser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    maskgaeparser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
    maskgaeparser.add_argument("--save_path", nargs="?", default="DPMGCDA.pt", help="save path for model.")
    maskgaeparser.add_argument("--device", type=int, default=0)

    return maskgaeparser.parse_args()



model_args = parse_args()
print(tab_printer(model_args))


set_seed(model_args.seed)
if model_args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{model_args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])

########################################################################

#data = get_dataset(root, args.dataset, transform=transform)
DPMGCDA_train(model_args,device)



