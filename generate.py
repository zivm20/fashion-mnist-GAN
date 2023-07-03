import torch
import argparse
import os
import src.config as cfg
from src.train_model import generate_samples
import cv2



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5, help="number of images to create")
    parser.add_argument("--folder", type=str, default="output", help="output folder path")
    parser.add_argument("--name", type=str, default="", help="name format of images")
    parser.add_argument("--model", type=str, default="checkpoints/G/final.pth", help="path to generator model")
    opt = parser.parse_args()
    
    



    model = torch.load(opt.model,map_location=torch.device(cfg.DEVICE))
    
    images = generate_samples(model,opt.num)

    for i,img in enumerate(images):
    
        cv2.imwrite(opt.folder+"/"+opt.name+str(i)+".jpg",img*255)
