from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import torch
import torch.nn as nn
import torch.nn.functional as F

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
