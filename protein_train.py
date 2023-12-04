import os
from torch.utils. data import Dataset, IterableDataset, DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn.functional as F
#from datasets import load_dataset
from accelerate import Accelerator
from protein_ae import Unet
from tqdm import tqdm as std_tqdm
from functools import partial
from glob import glob
import matplotlib.pyplot as plt
import random
from itertools import cycle,chain,islice
from torchvision.transforms import ToTensor
from torchvision import transforms
import tifffile as tf


#TODO: scale inputs and ouputs


def img_to_tiles(hist_img, img_dim, img_stride):
    res = []
    for i in range(0, hist_img.shape[0]-img_dim, img_stride):
        for j in range(0, hist_img.shape[1]-img_dim, img_stride):
            res.append(hist_img[i:i+img_dim,j:j+img_dim,])
    res = [np.moveaxis(x,-1,0) for x in res]
    return np.array(res)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accelerator = Accelerator()

# get data - normalize polyt/dapi inputs, predict raw tau, scale by 255
print("Fetching Data")
raw_data = np.moveaxis(tf.imread('2023_1108_dlpfc_tau_128totalframes_balanced.npy'),1,-1)[:30,:,:,:]
print("Tiling Data")

'''
p_max = np.max(raw_data[:,:,:,0])
raw_data[:,:,:,0] /= p_max
d_max = np.max(raw_data[:,:,:,1])
raw_data[:,:,:,1] /= d_max
t_max = np.max(raw_data[:,:,:,3])
raw_data[:,:,:,3] /= t_max

'''
for i,image in enumerate(raw_data):
    im_max = np.max(image[:,:,3])
    raw_data[i,:,:,3] /= im_max    
    poly_max = np.max(image[:,:,0])
    raw_data[i,:,:,0] /= poly_max
    dapi_max = np.max(image[:,:,1])
    raw_data[i,:,:,1] /= dapi_max


tiles = [img_to_tiles(x, 256, 128) for x in raw_data]
tiles = np.concatenate(tiles, axis=0)
filtered = []
for t in tiles:
    if (np.max(t[3,:,:]) < 0.1):
        continue
    else:
        filtered.append(t)
tiles = np.squeeze(np.array(filtered))
print(tiles.shape)
full_dataset = TensorDataset(torch.from_numpy(tiles))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=1)

print("Data Loaded")

model = Unet()


epochs = 10
lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
train_losses = []
test_losses = []

print("Starting Training")
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    for source in train_loader:
        optimizer.zero_grad()
        source = source[0].float()
        polyt_dapi = source[:,0:2,:,:]
        tau = source[:,3,:,:]

        output = model(polyt_dapi)
        pred = torch.squeeze(output)
        target = torch.squeeze(tau*255).long()
        loss = torch.nn.NLLLoss()(pred, target)

        accelerator.backward(loss)

        optimizer.step()

        train_loss += loss.item()
        torch.cuda.empty_cache()
    train_loss = train_loss/len(train_loader)
    print(f"Epoch: {epoch}, Training loss: {train_loss}")
    train_losses.append(train_loss)

    model.eval()
 
    with torch.no_grad():
        test_loss = 0.0
        for source in test_loader:
            source = source[0].float()
            polyt_dapi = source[:,0:2,:,:]
            tau = source[:,3,:,:]

            output = model(polyt_dapi)
            pred = torch.squeeze(output)
            target = torch.squeeze(tau*255).long()
            loss = torch.nn.NLLLoss()(pred, target)
            test_loss += loss.item()

            torch.cuda.empty_cache()
        test_loss /= len(test_loader)
        print(f"Testing loss: {test_loss}")
        test_losses.append(float(test_loss))

torch.save(model.state_dict(), 'out/protein_unet.pt')

i = 0
for source in test_loader:
    if i > 9:
        break
    i += 1
    batch = source[0].float()
    polyt_dapi = batch[:,0:2,:,:]
    tau = batch[:,3,:,:]

    output = model(polyt_dapi)
    pred = torch.squeeze(output[0])
    target = torch.squeeze(tau[0]*255).long()

    image = target.cpu().detach().numpy()
    image = np.moveaxis(image,0,-1)
    plt.imsave(f"out/true_protein{i}.png", image)

    inv_recon = pred.cpu().detach().numpy() 
    inv_recon = np.argmax(inv_recon, axis=0)
    inv_recon = np.moveaxis(inv_recon,0,-1)
    plt.imsave(f"out/recon_protein{i}.png", inv_recon)


epoch_list = [i for i in range(epochs)]
plt.plot(epoch_list, train_losses, label="Training")
plt.plot(epoch_list, test_losses, label="Testing")
plt.legend()
plt.savefig('out/protein_ae.png')

