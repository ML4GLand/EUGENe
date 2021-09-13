# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:09:42 2020

@author: james

InSNPtion.py builds a multi-task linear neural network. It is designed to take in 
relevant data and output unactivated outputs.  The reason for the unactivation is that it 
is assumed activation will be handled with sigmoid,softmax, etc and therefore is easier to do
in the training script when criterion are defined for each label and losses are grouped
together before stepping back. This takes advantage of the fact that pytorch built in loss
functions take in unactivated outputs (e.g., nn.CrossEntropyLoss()). For cases when this is 
not true (binary classification?) can always activate first relevant output and then take loss
function. This model architecture allows for toggling activation parameters between ReLU and Mish.

#NOTE: Using a batch size of 1 will throw an error due to pytorch's batch normalization expectations
so ensure running with batchsize >1. If want to handle batch size > 1 can adapt script to use instance norm 
in case when inputDimensions[0] == 1

"""

import torch.nn as nn
import torch.nn.functional as F
import torch

#Define a function to apply weight transforms 
def PhotoCopierInitialize(laya, default = "kaiming"):
    if isinstance(laya, nn.Linear) and default.lower() == "xavier":
        torch.nn.init.xavier_normal_(laya.weight.data)
        torch.nn.init.xavier_normal_(laya.bias.data.unsqueeze(0))
    elif isinstance(laya, nn.Linear) and default.lower() == "kaiming":
        #print("Initializing with Kaiming...")
        torch.nn.init.kaiming_normal_(laya.weight)
  
    #Can add other instantiations here as needed --> e.g. normal with .init.normal_  

def Mish(x):
    return (x*torch.tanh(F.softplus(x)))

class DiddyKongRacing(nn.Module):
    # INPUTS:
    # Input Dimensions are input here as "BATCH", number of SNPs; --> Did this way in case easier to pass in during training and potentially help in debugging --> can also just change to int
    # numLayers == The number of hidden layers in the network (i.e., before the multi-task learning application layers)
    # layerWidths == The widths of each layer (how many cells in each layer) #Type list 
    # dropout == fraction of nodes to dropout for regularization (default 0.5)
    # multitask outputs == the number of neurons in the final layer for each task (type list) --> defaulting to [1,1] for each for now (e.g, assuming a binary classification task for both)
    
    # OUTPUT: A list of unactivated ouputs 
    
    def __init__(self, inputDimensions = [2,512], numLayers = 3, layerWidths = None, dropout = 0.5, multitaskOutputs = [1,1], activation = "ReLU"):
        super(DiddyKongRacing,self).__init__()
        self.activation = activation
        self.getWide = None
        self.howManyLayers = numLayers
        if layerWidths == None:
            self.getWide = [inputDimensions[1]] + [256, 128, 64]
            
            '''
            Can do some auto-layer magic here based on mod2 for compression and number of layers layer size, etc, but for ffn w/o supporting literature/predefined 
            SNP/layer width and depth magic, leaving out for now 
            '''
            
        else:
            if numLayers != len(layerWidths):
                print("Hey idiot, do you not have any common sense? The number of layers needs to equal the layer widths. Because of your incompetence it will be handled automagically")
                print("\n")
            if layerWidths[0] > inputDimensions[1]: 
                print("Uhh why are you setting your layer widths higher than the number of elements????")
                print("\n")
            maximum = layerWidths[0]
            for el in layerWidths[1:]:
                if el > maximum:
                    print("Yo dumbass why are you setting your layer widths > the previous layer???? Don't do this!")
                maximum = el
                
            self.getWide = [inputDimensions[1]] + layerWidths[:numLayers]
            
            
                
        self.hotTopVolcano = nn.ModuleList() #hidden layers
        self.jetPacksAndPeanutPistols = nn.ModuleList() #multi-task output layers
        
        
        print("A ba ba boua ba ba: Layers are initializing...")
        for i in range(len(self.getWide) - 1):
            print("Adding linear layer from {} nodes to {} nodes...".format(self.getWide[i], self.getWide[i+1]))
            self.hotTopVolcano.append(DonkeyKong64(self.getWide[i],self.getWide[i+1], dropout, False, activation))
            
        #Add in multitask layers here
        print("The angry aztec sends his llama regards: Multi-task layers initializing...")
        for i in range(len(multitaskOutputs)):
            print("Adding multi-task linear layer from {} nodes to {} nodes".format(self.getWide[len(self.getWide)-1], multitaskOutputs[i]))
            self.jetPacksAndPeanutPistols.append(DonkeyKong64(self.getWide[len(self.getWide) - 1], multitaskOutputs[i], dropout, True, activation))
            
        
        
    def forward(self,x):
        #print("Foreword: forward or backward is just a matter of perspective")
        #print(x.size())
        for i in range(len(self.hotTopVolcano)):
            x = self.hotTopVolcano[i](x, self.activation, False)
        
        #Multi-task learning here... 
        blueShell  = []
        for el in self.jetPacksAndPeanutPistols:
            blueShell.append(el(x, self.activation, True))
        
        return blueShell
        
    
#Layer implementation: All layers use ReLU activation and employ batch normalization; Maybe LeakyReLU will be better??? easy to change...
class DonkeyKong64(nn.Module): 
    
    def __init__(self, diddyKong, donkeyKong, krool, itsTheFinalCountdown,krankyKong): #(self, input, output, dropout, isThisTheFinalLayerForMultiTask, layerActivation)
        super(DonkeyKong64, self).__init__()
        firingCoconuts = []
        if itsTheFinalCountdown:
            firingCoconuts = [nn.Linear(diddyKong, donkeyKong)]
        else:
            if krankyKong == "Mish":
                firingCoconuts = [nn.Linear(diddyKong, donkeyKong), nn.BatchNorm1d(donkeyKong), nn.Dropout(krool)]
                
            elif krankyKong == "LeakyReLU":
                print("Initializing to Leaky ReLU...\n")
                firingCoconuts = [nn.Linear(diddyKong, donkeyKong), nn.BatchNorm1d(donkeyKong), nn.Dropout(krool), nn.LeakyReLU(negative_slope = 0.2, inplace=True)]
            else:
                firingCoconuts = [nn.Linear(diddyKong, donkeyKong), nn.BatchNorm1d(donkeyKong), nn.Dropout(krool), nn.ReLU(inplace=True)]
            #print(firingCoconuts)
        
        self.goldBanana = nn.Sequential(*firingCoconuts)
        #print("Initializing Weights...")
        self.goldBanana.apply(PhotoCopierInitialize)
        
    def forward(self, x, activation, finalLayer):
        if (activation == "Mish") and (finalLayer == False):
            return Mish(self.goldBanana(x))
        else:
            return self.goldBanana(x)
        
  