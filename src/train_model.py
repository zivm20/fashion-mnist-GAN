from src import config as cfg
import torch
import matplotlib.pyplot as plt

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
    


def train(D:torch.nn.Module,G:torch.nn.Module,loader_train,loader_val, D_optimizer:torch.optim,G_optimizer:torch.optim,D_lossFunction=cfg.D_LOSS_FUNCTION,G_lossFunction=cfg.G_LOSS_FUNCTION, verbose = 0,save_checkpoints=None):
	
	G_hist = {"train_loss":[],"train_accuracy":[],"val_accuracy":[]}
	D_hist = {"train_loss":[],"train_accuracy":[],"val_accuracy":[]}
	img_hist = []
	
	G = G.to(device=cfg.DEVICE)  # move the model parameters to CPU/GPU
	D = D.to(device=cfg.DEVICE)
	
	for e in range(cfg.EPOCHES):
		if verbose>1:
			print('')
			
		for i,(x, y) in enumerate(loader_train):
			
			D.train()  # put model to training mode
			G.train()  # put model to training mode

			real_label = torch.ones(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)
			fake_label = torch.zeros(y.size(dim=0)).to(device=cfg.DEVICE,dtype=torch.long)


			#compute loss of D
			D_optimizer.zero_grad()
			
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
			img_hist.append(fake2.cpu().detach())
			G_hist["train_loss"].append(G_loss.item())
			G_loss.backward()
			
			#finally update G
			G_optimizer.step()

			
		
		train_acc_D,train_acc_G = check_accuracy(loader_train,D,G)
		val_acc_D,val_acc_G = check_accuracy(loader_val,D,G)
		
		D_hist["train_accuracy"].append(train_acc_D)
		D_hist["val_accuracy"].append(val_acc_D)

		G_hist["train_accuracy"].append(train_acc_G)
		G_hist["val_accuracy"].append(val_acc_G)


		
		if verbose>1:
			print('EPOCH %d, D loss: = %.4f, G loss:  %.4f' % (e, D_loss, G_loss))
			print('D train accuracy: (%.2f)%%, D val accuracy: (%.2f)%%' % (100 * train_acc_D,100*val_acc_D))
			print('G train accuracy: (%.2f)%%, G val accuracy: (%.2f)%%' % (100 * train_acc_G,100*val_acc_G))
		if verbose > 2:
			plt.imshow(img_hist[-1].numpy()[-1].transpose(1,2,0),cmap="gray")
			plt.show()
		if save_checkpoints != None:
			torch.save(D,"checkpoints/D/"+save_checkpoints+".pth")
			torch.save(D,"checkpoints/G/"+save_checkpoints+".pth")

	
	train_acc_D,train_acc_G = check_accuracy(loader_train,D,G)
	val_acc_D,val_acc_G = check_accuracy(loader_val,D,G)
	
	D_hist["train_accuracy"].append(train_acc_D)
	D_hist["val_accuracy"].append(val_acc_D)

	G_hist["train_accuracy"].append(train_acc_G)
	G_hist["val_accuracy"].append(val_acc_G)

	
	return (train_acc_D,train_acc_G,val_acc_D,val_acc_G,D_hist,G_hist,img_hist)