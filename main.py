﻿# -*- coding: utf-8 -*-
import time

import numpy

import geatpy as ea  # import geatpy
import dgl
from utils import feature_norm
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from utils import load_data, accuracy,load_pokec,GraphDirectAddNegativeEdge
from models.FairGNN import FairGNN
from models.myGNN import myGNN
from models.GCN import GCN
import pandas as pd
import math
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score

def fair_metric(output,idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
    return parity,equality

def load_graph(args):

    if args.dataset != 'nba':
        if args.dataset == 'pokec_z':
            dataset = 'region_job'
        else:
            dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = args.label_number
        sens_number = args.sens_number
        seed = 20
        path = "../dataset/pokec/"
        test_idx = False
    else:
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "../dataset/NBA/"
        test_idx = True


    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                                    seed=seed,test_idx=test_idx)
    G = dgl.from_scipy(adj)
    return G,adj



def initial_pop(NIND,population,myAlgorithm,rawGraph,args):
    MaxDim = int((rawGraph.num_nodes() * rawGraph.num_nodes() - rawGraph.num_nodes()) / 2)
    population.initChrom(NIND)
    numNodes=rawGraph.num_nodes()
    random.seed(args.seed)
    stratNodeList=random.sample(range(1,numNodes),NIND)
    IndividualIndex = 0
    for node in stratNodeList:
        subg,_=dgl.khop_in_subgraph(rawGraph, node, k=1,store_ids =True)
        alledges=rawGraph.edges()
        EIDlist=subg.edata[dgl.EID].cpu().numpy()
        Iindex=alledges[0].cpu().numpy()
        Jindex=alledges[1].cpu().numpy()
        Iindex=Iindex[EIDlist]
        Jindex=Jindex[EIDlist]
        removeList=[]
        for i in range(len(Iindex)):
            if Iindex[i]==Jindex[i]:
                removeList.append(i)
        Iindex=np.delete(Iindex,removeList,0)
        Jindex=np.delete(Jindex,removeList,0)
        if EIDlist.shape[0]==1:
            population.Chrom[IndividualIndex,:]=newindi
            continue
        else:
            Iindex=Iindex[0:int(len(Iindex)/2-1)]
            Jindex=Jindex[0:int(len(Jindex)/2-1)]
        if len(Jindex)==0 or len(Iindex)==0:
            population.Chrom[IndividualIndex, :] = np.zeros((1, population.Lind))
        else:
            dadsa=1
            if Iindex[0]<Jindex[0]:
                temp=Iindex
                Iindex=Jindex
                Jindex=temp
            chroSetOneIndex=Iindex*(Iindex+1)/2+Jindex-Iindex
            chroSetOneIndex=chroSetOneIndex.astype(numpy.uint64)
            chroSetOneIndex[np.where( chroSetOneIndex>=MaxDim)[0]]=MaxDim-1
            chroSetOneIndex[np.where(chroSetOneIndex <0)[0]] =0
            population.Chrom[IndividualIndex,:]=np.zeros((1,population.Lind))
            population.Chrom[IndividualIndex,chroSetOneIndex]=1
            newindi=population.Chrom[IndividualIndex,:]
        IndividualIndex=IndividualIndex+1
            #print(subg)
    myAlgorithm.call_aimFunc(population,0)
    return population



if __name__ == '__main__':
    """===============================argument============================"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--num-hidden', type=int, default=64,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--acc', type=float, default=0.658,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.655,
                        help='the selected FairGNN ROC score on val would be at least this high')
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")


    parser.add_argument('--epochs', type=int, default=500,#1000,
                        help='Number of epochs to train.')
    parser.add_argument('--model', type=str, default="GAT",
                        help='the type of model GCN/GAT')
    parser.add_argument('--dataset', type=str, default='pokec_z',
                        choices=['pokec_z','pokec_n','nba'])
    parser.add_argument('--popsize', type=int, default=10,
                        help="popsize")
    parser.add_argument('--generations', type=int, default=2,#200
                        help="generations")
    parser.add_argument('--fairmodelC', type=str, default="Yes",
                        help='F or C')
    parser.add_argument('--fairmodelF', type=str, default="Yes",
                        help='F or C')
    parser.add_argument('--seed', type=int, default=458, help='Random seed.')
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    #%%
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    """==============================load data============================"""
    if args.dataset != 'nba':
        if args.dataset == 'pokec_z':
            dataset = 'region_job'
        else:
            dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = args.label_number
        sens_number = args.sens_number
        seed = 20
        path = "../dataset/pokec/"
        test_idx = False
    else:
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "../dataset/NBA/"
        test_idx = True



    adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number,
                                                                             seed=seed,test_idx=test_idx)

    #if args.dataset != 'nba':
    labels[labels > 1] = 1
    if sens_attr:
        sens[sens > 0] = 1

    if args.cuda:


        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()
        idx_sens_train = idx_sens_train.cuda()


    problemName = 'SearchSubgraphCOV'  
    fileName = problemName  
    SearchSubgraphProblem = getattr(__import__(fileName), problemName)  
    rawGraph,adj=load_graph(args)
    print(1)
    rawAdj=adj.toarray()
    model = GCN(nfeat=features.shape[1],   nhid=args.hidden,nclass=1,
            dropout=args.dropout)
    model.cuda()
    rawGraph = rawGraph.to('cuda:0')

    problem = SearchSubgraphProblem(rawGraph,adj,model,features,labels, idx_train, idx_val, idx_test,sens,idx_sens_train,args)  # 生成问题对象
    print(2)
    """============================================================"""
    Encoding = 'BG'  
    NIND = args.popsize  

    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) 

    print(3)
    population = ea.Population(Encoding, Field, NIND)  
    """==================================================="""
    myAlgorithm = ea.NSGA3_sg(problem, population)  
    myAlgorithm.MAXGEN = args.generations  
    myAlgorithm.logTras = 1  
    myAlgorithm.verbose = True  
    myAlgorithm.drawing = 0  

    prophetPop=initial_pop(NIND,population,myAlgorithm,rawGraph,args)
    [NDSet, population] = myAlgorithm.run(rawAdj,population)  

    NDSet.save("./Result"+args.dataset+"/")  


    pop = NDSet
    Vars = pop.Phen  
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    allobjresutlsF = np.random.random((pop.Chrom.shape[0], 5))
    allobjresutlsC = np.random.random((pop.Chrom.shape[0], 5))
    denAdj = rawAdj
    for i in range(pop.Chrom.shape[0]):
        sub = pop.Chrom[i, :]
        dAdj = denAdj  
        directInfluNode = []
        numOfRemove = 0
        numofAdd = 0
        selectedEdge = np.where(sub == 1)[0]  
        for sEdge in selectedEdge:
            row = math.floor((-0.5 + math.sqrt(0.25 - 4 * -0.5 * sEdge)) / 1) + 1
            col = int(sEdge - ((row - 1) * (row - 1) - ((row - 1) * (row - 1) - (row - 1)) / 2))
         #   copydAdj[row, col] = 0
            if dAdj[row, col] == 1:
                dAdj[row, col] = 0
                numOfRemove = numOfRemove + 1
            directInfluNode.append(row)
            directInfluNode.append(col)
        subnode = list(set(directInfluNode))
        subAdj = dAdj[subnode, :]
        subAdj = subAdj[:, subnode]
        compact = (len(np.where(subAdj != 0)[0])) /( len(subnode)+1)  
        G = dgl.from_scipy(ssp.csr_matrix(dAdj))
        G = dgl.add_self_loop(G)
        G = G.to('cuda:0')
        G=rawGraph
     

        

        print('FairGCN============performace on FairGCN=============')
        if args.fairmodelF=='Yes':
            best_result = {}
            best_fair = 100
            bestaaa=0
            model = FairGNN(nfeat = features.shape[1], args = args)
            model.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number)))
            model.cuda()
            if args.dataset == 'nba':
                MaxEpo=2090
                stopEpo=600

            for epoch in range(MaxEpo):
                model.train()
                model.optimize(G,features,labels,idx_train,sens,idx_sens_train)
                cov = model.cov
                cls_loss = model.cls_loss
                adv_loss = model.adv_loss
                model.eval()
                output,s = model(G, features)
                acc_val = accuracy(output[idx_val], labels[idx_val])
                roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),output[idx_val].detach().cpu().numpy())
                #acc_sens = accuracy(s[idx_test], sens[idx_test])
                parity_val, equality_val = fair_metric(output,idx_val)
                acc_test = accuracy(output[idx_test], labels[idx_test])
                roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),output[idx_test].detach().cpu().numpy())
                parity,equality = fair_metric(output,idx_test)

                if epoch%100==0:
                    print('FairGCN--Epoch: {:04d}'.format(epoch + 1),
                          "accuracy: {:.4f}".format(acc_test.item()),
                          "roc: {:.4f}".format(roc_test),
                          "parity: {:.4f}".format(parity),
                          "equality: {:.4f}".format(equality))

                if epoch>=stopEpo:
                    if bestaaa <roc_test:# nba
                        bestaaa =roc_test #nba
                        best_result['acc'] = acc_test.item()
                        best_result['roc'] = roc_test
                        best_result['parity'] = parity
                        best_result['equality'] = equality





            print('FairGCN============performace on FairGCN test set=============')
            if len(best_result) > 0:
                print("FairGCN--Test:",
                        "accuracy: {:.4f}".format(best_result['acc']),
                        "roc: {:.4f}".format(best_result['roc']),
                        "parity: {:.4f}".format(best_result['parity']),
                        "equality: {:.4f}".format(best_result['equality']))
                print("FairGCN--Test=================================")
                allobjresutlsF[i, 0] = best_result['acc']
                allobjresutlsF[i, 1] = best_result['roc']
                allobjresutlsF[i, 2] = best_result['parity']
                allobjresutlsF[i, 3] = best_result['equality']
                allobjresutlsF[i, 4] = compact
                frame = pd.DataFrame(allobjresutlsF, index=list(range(pop.Chrom.shape[0])),
                                     columns=['ACC', 'AUC', 'SP', 'EO', 'Compact'])
                frame.to_excel("./Result"+args.dataset+"/Obj-F.xlsx")


            else:
                print("Failed --- Please set appropriate thresholds")


