import torch
from torch import nn
import numpy as np
from src import config as cfg


class Generator(nn.Module):
    def __init__(self,noise_dim,hidden_dims=[1024,128,128,64,32,1]):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(noise_dim,hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                
                nn.Linear(hidden_dims[0],7*7*hidden_dims[1]),
                nn.BatchNorm1d(7*7*hidden_dims[1]),
                nn.ReLU(),


                nn.Unflatten(1,(hidden_dims[1],7,7)), #7x7
                
                
                nn.ConvTranspose2d(hidden_dims[1],hidden_dims[2],(5,5),stride=2,padding=2,output_padding=1), #14x14
                nn.BatchNorm2d(hidden_dims[2]),
                nn.ReLU(),

                nn.Conv2d(hidden_dims[2],hidden_dims[3],(5,5),stride=1,padding=2), #14x14
                nn.BatchNorm2d(hidden_dims[3]),
                nn.ReLU(),

                nn.ConvTranspose2d(hidden_dims[3],hidden_dims[4],(5,5),stride=2,padding=2,output_padding=1), #28x28
                nn.BatchNorm2d(hidden_dims[4]),
                nn.ReLU(),

                nn.Conv2d(hidden_dims[4],hidden_dims[5],(5,5),stride=1,padding=2), #28x28
                nn.BatchNorm2d(hidden_dims[5]),
                nn.ReLU(),

                nn.Tanh()#output in the range (-1,1), since the input is normalized
            )
       
        
    def forward(self,x):
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, in_shape=(28,28,1),out_classes=2,dropout=0.2, hidden_dims=[32,64,128,1024]):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_shape[-1],hidden_dims[0],kernel_size=(5,5),stride=2,padding=2), #14x14
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),


            nn.Conv2d(hidden_dims[0],hidden_dims[1],kernel_size=(5,5),stride=2,padding=2), #7x7
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),


            nn.Conv2d(hidden_dims[1],hidden_dims[2],kernel_size=(5,5),stride=1,padding=2), #7x7
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),

            
            nn.Flatten(),


            nn.Linear(hidden_dims[2]*7*7,hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[3],out_classes),
            nn.Sigmoid()#binary classification
        )

    def forward(self,x):
        return self.model(x)
    



    


