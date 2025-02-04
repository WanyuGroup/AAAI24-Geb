# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils import accuracy
import math
import dgl
import scipy.sparse as ssp
import os
import time
import pandas as pd
from sklearn.metrics import roc_auc_score



class SearchSubgraphCOV(ea.Problem):  
    def __init__(self, rawGraph, adj, model, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train,
                 args):
        name = 'SearchSubgraph'  
        M = 3  
        self.rawG = rawGraph
        self.sparseAdj = adj
        stime=time.time()
        self.denAdj =adj.toarray()
        etime=time.time()
        print(etime-stime)
        self.args = args
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.idx_sens_train = idx_sens_train
        self.rawModel = model
        self.M = M
        maxormins = [1] * M  
        Dim = int((rawGraph.num_nodes() * rawGraph.num_nodes() - rawGraph.num_nodes()) / 2)  
        varTypes = [1] * Dim  
        lb = [0] * Dim  
        ub = [1] * Dim  
        lbin = [1] * Dim 
        ubin = [1] * Dim  
        self.rawGNN_SP, self.rawGNN_EO = self.trainModel(model)
  
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def fair_metric(self, output, idx):
        labels = self.labels
        sens = self.sens
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)
        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))
        return parity, equality

    def updatePartialModel(self,G,newAdj,directInfluNode):
        args=self.args
        model=self.model
        H=self.H.clone()
        maxH,_=H.max(0)
        minH,_=H.min(0)
        idx_train=self.idx_train
        features=self.features
        labels=self.labels
        idx_sens_train=self.idx_sens_train
        sens=self.sens
        idx_test=self.idx_test
        idx_val=self.idx_val
        best_acc = 0.0
        best_test = 0.0
        allInfluNode=[]
        for node in directInfluNode:
            hop1neighor=list(set(newAdj[node]))
            flufactorL1=1/len(hop1neighor)
            for hop1node in hop1neighor:
                hop1node=int(hop1node)
                allInfluNode.append(hop1node)
                hop2neighor=list(set(newAdj[hop1node]))
                flufactorL2=1/len(hop2neighor)
                flufactorL1=(flufactorL1+flufactorL2)/len(hop2neighor)
                H[hop1node]=F.softmax(H[hop1node]-flufactorL1*1/len(directInfluNode)*H[node],dim=-1)
                for hop2node in hop2neighor:
                    hop2node=int(hop2node)
                    allInfluNode.append(hop2node)
                    H[hop2node]=F.softmax(H[hop2node]-flufactorL2*1/len(directInfluNode)*H[node],dim=-1)


        allInfluNode=list(set(allInfluNode))
        #loss_Influ = F.binary_cross_entropy_with_logits(H[allInfluNode], self.H[allInfluNode].float())
        #deleteNodeList=np.where(np.sum(newAdj,axis=0)==0)[0]
        optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            output = model(G, features)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
            loss_Influ=F.binary_cross_entropy_with_logits(output[allInfluNode],H[allInfluNode].detach().float())
            loss_train=loss_train+0.1*loss_Influ
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(G, features)
            if epoch%2==0:
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_test = accuracy(output[idx_test], labels[idx_test])
                parity_val, equality_val = self.fair_metric(output,idx_val)
                parity,equality = self.fair_metric(output,idx_test)
                roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())


                print("modify Graph Epoch [{}] Test set results:".format(epoch),
                    "acc_test= {:.4f}".format(acc_test.item()),
                    "acc_val: {:.4f}".format(acc_val.item()),
                      "roc: {:.4f}".format(roc_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality),
                     "loss_Influ: {:.4f}".format(loss_Influ),
                       "loss_train: {:.4f}".format(loss_train),
                      )
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    #torch.save(model.state_dict(),"./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number))
                if acc_val.unsqueeze(0).cpu().numpy()[0] > 0.65 and parity_val < 0.05 and parity_val > 0:
                    return parity, equality, acc_test, roc_test
                elif epoch > 8 and acc_val.unsqueeze(0).cpu().numpy()[0] > 0.63 and parity_val < 0.07:
                    return parity, equality, acc_test, roc_test
        return parity,equality,acc_test,roc_test

    def updateModel(self, G):
        args = self.args
        model = self.model
        features = self.features
        idx_sens_train = self.idx_sens_train
        sens = self.sens
        idx_test = self.idx_test
        idx_val = self.idx_val
        best_acc = 0.0
        best_test = 0.0
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(G, features)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train],
                                                            sens[idx_sens_train].unsqueeze(1).float())
            acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(G, features)
            if epoch % 10 == 0:
                acc_val = accuracy(output[idx_val], sens[idx_val])
                acc_test = accuracy(output[idx_test], sens[idx_test])
                parity_val, equality_val = self.fair_metric(output, idx_val)
                parity, equality = self.fair_metric(output, idx_test)
                roc_test = roc_auc_score(sens[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())

                print("modify Graph Epoch [{}] Test set results:".format(epoch),
                      "acc_test= {:.4f}".format(acc_test.item()),
                      "roc: {:.4f}".format(roc_test),
                      "acc_val: {:.4f}".format(acc_val.item()),
                      "parity: {:.4f}".format(parity),
                      "equality: {:.4f}".format(equality))
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    # torch.save(model.state_dict(),"./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number))
        return parity, equality, acc_test

    def trainModel(self, model):
        args = self.args
        features = self.features
        idx_sens_train = self.idx_sens_train
        sens = self.sens
        idx_test = self.idx_test
        idx_val = self.idx_val
        best_acc = 0.0
        best_test = 0.0
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(self.rawG, features)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_sens_train],
                                                            sens[idx_sens_train].unsqueeze(1).float())
            # acc_train = accuracy(output[idx_sens_train], sens[idx_sens_train])
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(self.rawG, features)
            if epoch % 10 == 0:
                acc_val = accuracy(output[idx_val], sens[idx_val])
                acc_test = accuracy(output[idx_test], sens[idx_test])
                parity_val, equality_val = self.fair_metric(output, idx_val)
                parity, equality = self.fair_metric(output, idx_test)
                roc_test = roc_auc_score(sens[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())


                print("Graph Epoch [{}] Test set results:".format(epoch),
                      "acc_test= {:.4f}".format(acc_test.item()),
                      "roc: {:.4f}".format(roc_test),
                      "acc_val: {:.4f}".format(acc_val.item()),
                      "parity: {:.4f}".format(parity),
                      "equality: {:.4f}".format(equality))
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    # torch.save(model.state_dict(),"./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number))
        self.model = model
        self.H = output
        return parity, equality

    def aimFunc(self, pop,genI):
        Vars = pop.Phen 
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        pop.ObjV = np.zeros((pop.Chrom.shape[0], self.M))
        allobjresutls=np.random.random((pop.Chrom.shape[0],5))
        # adasdsa=Vars[:,[0,1]]
        for i in range(pop.Chrom.shape[0]):
            sub = pop.Chrom[i, :]
            selectedEdge = np.where(sub == 1)[0]  
            dAdj = self.denAdj  
            directInfluNode = []
            for sEdge in selectedEdge:
                row = math.floor((-0.5 + math.sqrt(0.25 - 4 * -0.5 * sEdge)) / 1) + 1
                col = int(sEdge - ((row - 1) * (row - 1) - ((row - 1) * (row - 1) - (row - 1)) / 2))
                dAdj[row, col] = 0  
                directInfluNode.append(row)
                directInfluNode.append(col)
            subnode=list(set(directInfluNode))
            subAdj=dAdj[subnode,:]
            subAdj=subAdj[:,subnode]
            compact=(len(np.where(subAdj!=0)[0]))/(len(subnode)+1)
            G=dgl.from_scipy(ssp.csr_matrix(dAdj))
            G = dgl.add_self_loop(G)
            G = G.to('cuda:0')
            directInfluNode=list(set(directInfluNode))
            sp,eo,acc,roc=self.updatePartialModel(G,dAdj,directInfluNode)
            acc=acc.unsqueeze(0).cpu().numpy()[0]
            allobjresutls[i,0]=acc
            allobjresutls[i,1]=roc
            allobjresutls[i,2]=sp
            allobjresutls[i,3]=eo
            allobjresutls[i,4]=compact
            pop.ObjV[i,:]=[sp,-acc,compact]
        
        frame = pd.DataFrame(allobjresutls,index=list(range(pop.Chrom.shape[0])),columns=['ACC','AUC','SP','EO','Compact'])
        frame.to_excel("./Result"+self.args.dataset+"/Obj"+str(genI)+".xlsx")

      
