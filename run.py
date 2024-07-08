### Import Necessary Libraries ###
import pickle
import timeit
import os
import random
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

### Check if GPU is available ###
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### Define k-folds ###
num_kfolds = 5
kfold = KFold(n_splits=num_kfolds, shuffle=True, random_state=1)


import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch


class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint, dim, layer_gnn, esm_dim):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.linear1 = nn.Linear(7, dim)
        self.linear2 = nn.Linear(7, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_gnn1 = nn.Linear(dim, dim)
        self.W_gnn2 = nn.Linear(dim, dim)
        self.conv1d1 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.conv1d2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.biGRU1 = nn.GRU(20, 10, bidirectional=True, batch_first=True, num_layers=1)
        self.biGRU2 = nn.GRU(20, 10, bidirectional=True, batch_first=True, num_layers=1)
        self.global_avgpool1d1 = nn.AdaptiveAvgPool1d(20)
        self.global_avgpool1d2 = nn.AdaptiveAvgPool1d(20)
        self.W1_attention = nn.Linear(dim, dim)
        self.W2_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(2 * dim, 2)  # Change the input dimension to 2 * dim

        # self.lambda_fusion = nn.Parameter(torch.tensor(0.5))
        # self.lambda_fusion = 0.1、0.2、0.3、0.5、0.7、1
        self.lambda_fusion = torch.tensor(0.3)

        # Additional ESM feature processing layers
        self.W_esm = nn.Linear(esm_dim, dim)

    def gnn(self, xs1, A1, xs2, A2):
        for i in range(len(self.W_gnn)):
            xs1 = xs1.transpose(0, 1)
            xs1 = self.conv1d1(xs1)
            xs1 = xs1.transpose(0, 1)
            # print(xs1.shape)
            xs1, _ = self.biGRU1(xs1)
            # print(xs1.shape)
            xs1 = self.global_avgpool1d1(xs1)
            # print(xs1.shape)
            xs2 = xs2.transpose(0, 1)
            xs2 = self.conv1d2(xs2)
            xs2 = xs2.transpose(0, 1)
            xs2, _ = self.biGRU2(xs2)
            xs2 = self.global_avgpool1d2(xs2)
            hs1 = torch.relu(self.W_gnn[i](xs1))
            hs2 = torch.relu(self.W_gnn[i](xs2))
            xs1 = torch.matmul(A1, hs1)
            xs2 = torch.matmul(A2, hs2)

        return xs1, xs2

    def dot_product_attention(self, h1, h2):
        # Dot-product attention mechanism
        d_k = dim
        scaling_factor = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_scores1 = torch.matmul(h1, h2.transpose(-1, -2)) / scaling_factor
        attention_scores2 = attention_scores1.transpose(-1, -2)
        attention_probs1 = torch.softmax(attention_scores1, dim=-1)
        attention_probs2 = torch.softmax(attention_scores2, dim=-1)
        attn_output1 = torch.matmul(attention_probs1, h2)
        attn_output2 = torch.matmul(attention_probs2, h1)

        s1 = torch.mean(attn_output1, dim=0).unsqueeze(0)
        s2 = torch.mean(attn_output2, dim=0).unsqueeze(0)
        return torch.cat((s1, s2), dim=1)

    def forward(self, inputs):
        adjacency1, adjacency2, esm_feature1, esm_feature2, protein1, protein2 = inputs

        # Embedding ESM features
        esm1 = torch.relu(self.W_esm(esm_feature1)).unsqueeze(0)
        esm2 = torch.relu(self.W_esm(esm_feature2)).unsqueeze(0)

        #residue feature
        protein1 = self.linear1(protein1)
        protein2 = self.linear2(protein2)

        # GNN processing
        x_protein1, x_protein2 = self.gnn(protein1, adjacency1, protein2, adjacency2)


        # Feature fusion
        lambda_fusion = torch.sigmoid(self.lambda_fusion)
        x_protein1 = lambda_fusion * esm1 + (1 - lambda_fusion) * x_protein1
        x_protein2 = lambda_fusion * esm2 + (1 - lambda_fusion) * x_protein2


        # Mutual attention
        y = self.dot_product_attention(x_protein1, x_protein2)

        # Predict interaction
        z_interaction = self.W_out(y)

        return z_interaction

    def __call__(self, data, train=True):
        inputs, t_interaction = data[:-1], data[-1]
        z_interaction = self.forward(inputs)
        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
            t = int(t_interaction.to('cpu').data[0].numpy())
            return z, t


### Define Trainer Class ###

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):

        sampling  = random.choices(dataset, k=1000)

        loss_total = 0
        for data in sampling:

            s1, i1 = file_ind(data[0])
            s2, i2 = file_ind(data[1])

            A1 = np.load(dir_input+'adjacencies_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            A2 = np.load(dir_input+'adjacencies_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)

            E1 = np.load(dir_input+'esm_features_batch_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            E2 = np.load(dir_input+'esm_features_batch_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)

            D1 = np.load(dir_input+'batch_residue_features_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            D2 = np.load(dir_input+'batch_residue_features_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)


            protein11 = torch.FloatTensor(D1[i1])
            protein22 = torch.FloatTensor(D2[i2])

            esm1 = torch.FloatTensor(E1[i1])
            esm2 = torch.FloatTensor(E2[i2])

            adjacency1 = torch.FloatTensor(A1[i1])
            adjacency2 = torch.FloatTensor(A2[i2])

            interaction = torch.LongTensor([data[2]])

            comb = (adjacency1.to(device), adjacency2.to(device),
                    esm1.to(device), esm2.to(device), protein11.to(device), protein22.to(device), interaction.to(device))

            loss = self.model(comb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


### Define Tester Class ###

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):

        sampling = dataset

        z_list, t_list = [], []
        for data in sampling:

            s1, i1 = file_ind(data[0])
            s2, i2 = file_ind(data[1])

            A1 = np.load(dir_input+'adjacencies_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            A2 = np.load(dir_input+'adjacencies_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)

            E1 = np.load(dir_input+'esm_features_batch_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            E2 = np.load(dir_input+'esm_features_batch_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)

            D1 = np.load(dir_input+'batch_residue_features_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            D2 = np.load(dir_input+'batch_residue_features_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)


            protein11 = torch.FloatTensor(D1[i1])
            protein22 = torch.FloatTensor(D2[i2])

            esm1 = torch.FloatTensor(E1[i1])
            esm2 = torch.FloatTensor(E2[i2])

            adjacency1 = torch.FloatTensor(A1[i1])
            adjacency2 = torch.FloatTensor(A2[i2])

            interaction = torch.LongTensor([data[2]])

            comb = (adjacency1.to(device), adjacency2.to(device),
                    esm1.to(device), esm2.to(device), protein11.to(device), protein22.to(device), interaction.to(device))

            z, t = self.model(comb, train=False)
            z_list.append(z)
            t_list.append(t)

        score_list, label_list = [], []
        for z in z_list:
            score_list.append(z[1])
            label_list.append(np.argmax(z))

        labels = np.array(label_list)
        y_true = np.array(t_list)
        y_pred = np.array(score_list)

        tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv = calculate_performace(len(sampling), labels,  y_true)
        roc_auc_val = roc_auc_score(t_list, score_list)
        fpr, tpr, thresholds = roc_curve(labels, y_pred) #probas_[:, 1])
        auc_val = auc(fpr, tpr)

        return accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn

    def result(self, epoch, time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_name):
        with open(file_name, 'a') as f:
            result = map(str, [epoch, time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn])
            f.write('\t'.join(result) + '\n')

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)


### Utility functions ###

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def file_ind(index):
    st_ind, in_ind = divmod(index,10)
    return 10*st_ind, in_ind

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1


    if (tp+fn) == 0:
        q9 = float(tn-fp)/(tn+fp + 1e-06)
    if (tn+fp) == 0:
        q9 = float(tp-fn)/(tp+fn + 1e-06)
    if  (tp+fn) != 0 and (tn+fp) !=0:
        q9 = 1- float(np.sqrt(2))*np.sqrt(float(fn*fn)/((tp+fn)*(tp+fn))+float(fp*fp)/((tn+fp)*(tn+fp)))

    Q9 = (float)(1+q9)/2
    accuracy = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    recall = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    ppv = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    F1_score = float(2*tp)/(2*tp + fp + fn + 1e-06)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

    return tp,fp,tn,fn,accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv


### Hyperparameters ###

radius         = 1
dim            = 20
layer_gnn      = 3
lr             = 1e-3
lr_decay       = 0.5
decay_interval = 10
iteration      = 50
esm_dim = 1280


### Dataset preparation and training ###

dir_input        = ('pdb_files/input'+str(radius)+'/')
examples         = np.load(dir_input + 'balanced_train')
fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
n_fingerprint = len(fingerprint_dict) + 100


fold_count = 1
for train, test in kfold.split(examples):
    dataset_train = examples[train]
    dataset_test  = examples[test]

    torch.manual_seed(1234)
    model = ProteinProteinInteractionPrediction(n_fingerprint, dim, layer_gnn, esm_dim).to(device)
    trainer = Trainer(model)
    tester  = Tester(model)

    file_result = 'run_output/result/one/' + 'results_fold_' + str(fold_count) + '.txt'
    os.makedirs('run_output/result/one/', exist_ok=True)
    with open(file_result, 'w') as f:
        f.write('Epoch \t Time(sec) \t Loss_train \t Accuracy \t Precision \t Recall \t Sensitivity \t Specificity \t MCC \t F1-score \t ROC_AUC \t AUC \t Q9 \t PPV \t NPV \t TP \t FP \t TN \t FN\n')

    file_model = 'run_output/model/one/' + 'model_fold_' + str(fold_count)
    os.makedirs('run_output/model/one/', exist_ok=True)

    print('Training...')
    start = timeit.default_timer()

    for epoch in range(iteration):
        if (epoch+1) % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss = trainer.train(dataset_train)

        accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn = tester.test(dataset_test)

        end  = timeit.default_timer()
        time = end - start

        tester.result(epoch, time, loss,  accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_result)
        tester.save_model(model, file_model)

        print(f"Model saved to {file_model}")

        # ...

        # 加载模型
        model.load_state_dict(torch.load(file_model, map_location=device))
        print(f"Model loaded from {file_model}")

        print('Epoch: ' + str(epoch))
        print('Accuracy: ' + str(accuracy))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Sensitivity: ' + str(sensitivity))
        print('Specificity: ' + str(specificity))
        print('MCC: ' + str(MCC))
        print('F1-score: ' + str(F1_score))
        print('ROC-AUC: ' + str(roc_auc_val))
        print('AUC: ' + str(auc_val))
        print('Q9: ' + str(Q9))
        print('PPV: ' + str(ppv))
        print('NPV: ' + str(npv))
        print('TP: ' + str(tp))
        print('FP: ' + str(fp))
        print('TN: ' + str(tn))
        print('FN: ' + str(fn))
        print('\n')


    fold_count += 1



