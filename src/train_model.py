from src import config as cfg
import torch
import matplotlib.pyplot as plt
import time
from src.model import Generator
import numpy as np

def check_accuracy(loader,D,G):
	
	num_correct = 0
	num_samples = 0
	num_inccorect = 0

	D.eval()  # set model to evaluation mode
	G.eval()
	
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=cfg.DEVICE)  # move to device, e.g. GPU
			
			scores_D = D(x)
			real_label = torch.ones(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)
			fake_label = torch.zeros(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)
			
			_, preds = scores_D.max(1)
			
			num_correct += (preds == real_label).sum()
			num_samples += preds.size(0)

			noise = torch.randn(y.size(dim=0),cfg.NOISE_DIM,device=cfg.DEVICE,dtype=torch.float32)
			scores_D = D(G(noise))
			
			_, preds = scores_D.max(1)
			num_correct += (preds == fake_label).sum()
			num_inccorect += (preds == real_label).sum()
			num_samples += preds.size(0)

		acc_D = float(num_correct) / num_samples
		acc_G = float(num_inccorect) / (num_samples/2)
	return acc_D,acc_G


def generate_samples(G:Generator,amount=1):
	noise = torch.randn(amount,cfg.NOISE_DIM,device=cfg.DEVICE,dtype=torch.float32)
	G.eval()
	with torch.no_grad():
		return G(noise).cpu().detach().numpy().transpose(0,2,3,1)
    


def train(D:torch.nn.Module,G:torch.nn.Module,loader_train,loader_val, D_optimizer:torch.optim,G_optimizer:torch.optim,D_lossFunction=cfg.D_LOSS_FUNCTION,G_lossFunction=cfg.G_LOSS_FUNCTION, verbose = 0,save_checkpoints=None,eval_every=10,saveSamples=2):
	
	G_hist = {"train_loss":[],"train_accuracy":[],"val_accuracy":[]}
	D_hist = {"train_loss":[],"train_accuracy":[],"val_accuracy":[]}
	img_hist = []
	
	G = G.to(device=cfg.DEVICE)  # move the model parameters to CPU/GPU
	D = D.to(device=cfg.DEVICE)
	
	for e in range(cfg.EPOCHES):
		if verbose>1:
			print('')
		start = time.time()
		for i,(x, y) in enumerate(loader_train):

			D.train()  # put model to training mode
			G.train()  # put model to training mode

			real_label = torch.ones(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)
			fake_label = torch.zeros(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)


			#compute loss of D
			D_optimizer.zero_grad()
			G_optimizer.zero_grad()
			
			#train Discriminator on a real data
			x = x.to(device=cfg.DEVICE,dtype=torch.float32)  # move to device, e.g. GPU
			D_scores = D(x)
			D_loss_real = D_lossFunction(D_scores, real_label) 

			


			#train Discriminator on fake samples
			noise = torch.randn(y.size(dim=0),cfg.NOISE_DIM,device=cfg.DEVICE,dtype=torch.float32)
			fake = G(noise)
			D_scores2 = D(fake)
			D_loss_fake = D_lossFunction(D_scores2, fake_label) 


			#full loss for D
			D_loss = (D_loss_fake+D_loss_real)
			D_hist["train_loss"].append(D_loss.item())
			D_loss.backward()
			
			# finally update D
			D_optimizer.step()

			
			real_label = torch.ones(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)
			
			
			#compute loss of G
			G_optimizer.zero_grad()
			
			#create a fake sample
			fake2 = G(noise)
			D_scores3 = D(fake2)
			#G succeeds when it fools D 
			G_loss = G_lossFunction(D_scores3,real_label)
			
			#save paramaters
			G_hist["train_loss"].append(G_loss.item())
			G_loss.backward()

			#finally update G
			G_optimizer.step()

		

		if verbose>1:
			print('EPOCH %d' % (e))
			print('D loss: = %.4f, G loss:  %.4f' % (D_loss, G_loss))
				


		if e%eval_every == 0:

			train_acc_D,train_acc_G = check_accuracy(loader_train,D,G)
			val_acc_D,val_acc_G = check_accuracy(loader_val,D,G)
			
			D_hist["train_accuracy"].append(train_acc_D)
			D_hist["val_accuracy"].append(val_acc_D)

			G_hist["train_accuracy"].append(train_acc_G)
			G_hist["val_accuracy"].append(val_acc_G)


		
			if verbose>1:
				print('D train accuracy: %.2f%%, G train accuracy: (%.2f)%%' % (100 * train_acc_D,100*train_acc_G))
				print('D val accuracy: %.2f%%, G val accuracy: (%.2f)%%' % (100 * val_acc_D,100*val_acc_G))
			if verbose > 2 and saveSamples>0:
				img_hist.append(generate_samples(G,saveSamples**2))
				fig, axes = plt.subplots(saveSamples, saveSamples)
				fig.set_size_inches(3*saveSamples,3*saveSamples)
				
				for i in range(saveSamples):
					for j in range(saveSamples):
						axes[i,j].imshow(img_hist[-1][i*saveSamples + j],cmap="gray")
				plt.show()


			if save_checkpoints != None:
				torch.save(D,"checkpoints/D/"+save_checkpoints+".pth")
				torch.save(D,"checkpoints/G/"+save_checkpoints+".pth")

		print("epoch time:",time.time()-start)

		
	train_acc_D,train_acc_G = check_accuracy(loader_train,D,G)
	val_acc_D,val_acc_G = check_accuracy(loader_val,D,G)
	
	D_hist["train_accuracy"].append(train_acc_D)
	D_hist["val_accuracy"].append(val_acc_D)

	G_hist["train_accuracy"].append(train_acc_G)
	G_hist["val_accuracy"].append(val_acc_G)

	img_hist = np.array(img_hist)
	img_hist = img_hist.reshape((img_hist.shape[0]*img_hist.shape[1],img_hist.shape[2],img_hist.shape[3]))

	
	return (train_acc_D,train_acc_G,val_acc_D,val_acc_G,D_hist,G_hist,img_hist)