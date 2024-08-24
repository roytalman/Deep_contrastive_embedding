from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive import Contrastive_loss

def plot_tsne(data, labels, title='T-sne'):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, random_state=2)
    z = tsne.fit_transform(data)
    df = pd.DataFrame(columns=['t-sne1', 't-sne2', 'label', 'size'])
    df['t-sne1'] = z[:, 0]
    df['t-sne2'] = z[:, 1]
    df['label'] = labels
    
    # Assign sizes; create a new column for sizes
    df['size'] = np.ones(df.shape[0]) * 70  # Default size
    df.loc[df['label'] == 1, 'size'] = 100  # Increase size for label==1
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='t-sne1', y='t-sne2',
        hue="label",
        size="size",  # Use the 'size' column for varying sizes
        sizes=(70, 100),  # Specify the range of sizes
        data=df,
        legend="full",
        palette="deep"
    )
    
    plt.title(title)
    plt.show()
    
    
class Net_embed(nn.Module):
    # A  fully connected NN for new finetune embedding learning:
    def __init__(self,input_dim = 768,hidden_dim = 128,out_dim = 16,drop_prob = 0.3):
        super(Net_embed, self).__init__()
        
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(drop_prob)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.drop(x)
        x = self.drop(self.act(self.bn(self.fc1(x))))
        #x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
def finetune_embeddig(embedding_mat_train, y_train, save_path =[], embedding_mat_test =[],
                      hidden_dim=512, out_dim = 64, drop_prob=0.3, margin = 0.2, learning_rate=0.001,
                     N_epoch= 10, batches_per_epoch= 750, batch_size= 32, verbose = True):
    
    # Initialize the network
    net = Net_embed(input_dim=embedding_mat_train.shape[1],
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    drop_prob=drop_prob)
    
    # Set hyperparameters:
    N_data = embedding_mat_train.shape[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    embedding_train_torch = torch.from_numpy(embedding_mat_train).float()
    label_train_torch = torch.from_numpy(y_train)
    
    criterion = Contrastive_loss(margin=margin)
    for n in range(N_epoch):
        batch_loss = 0
        for i in range(batches_per_epoch):
            optimizer.zero_grad()
            data_samp = np.random.choice(N_data, batch_size, replace=False)
            data_b = embedding_train_torch[data_samp, :]
            labels_b = label_train_torch[data_samp]
            pred = net(data_b)
            loss = criterion(pred, labels_b)
            loss.backward()
            optimizer.step()
            batch_loss += float(loss)
        if verbose:
            print(f"Epoch: {n}, loss: {batch_loss:.3f}")
        
    net.eval()
    # save NN
    if len(save_path):
        torch.save(net,save_path)
    # Predict data (notice that for large dataset it should be done iterativly )
    if len(embedding_mat_test):
        embedding_train_finetuned = net(embedding_mat_train)
        embedding_test_finetuned = net(embedding_mat_test)
        return embedding_train_finetuned, embedding_test_finetuned, net
    else:
        return net
