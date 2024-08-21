import argparse
import numpy as np
import torch

from utils import Net_embed

# Define the main function with argparse
def main(args):
    # Load the embeddings and labels
    embedding_mat_train = np.load(args.embedding_path)
    y_train = np.load(args.labels_path)
    
    # Initialize the network
    net = Net_embed(input_dim=embedding_mat_train.shape[1],
                    hidden_dim=args.hidden_dim,
                    out_dim=args.out_dim,
                    drop_prob=args.drop_prob)
    
    N_epoch = args.epochs
    batches_per_epoch = args.batches_per_epoch
    N_samp_batch = args.batch_size
    N_data = embedding_mat_train.shape[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    net.train()
    embedding_train_torch = torch.from_numpy(embedding_mat_train).float()
    label_train_torch = torch.from_numpy(y_train)
    
    criterion = Contrastive_loss(margin=args.margin)
    for n in range(N_epoch):
        batch_loss = 0
        for i in range(batches_per_epoch):
            optimizer.zero_grad()
            data_samp = np.random.choice(N_data, N_samp_batch, replace=False)
            data_b = embedding_train_torch[data_samp, :]
            labels_b = label_train_torch[data_samp]
            pred = net(data_b)
            loss = criterion(pred, labels_b)
            loss.backward()
            optimizer.step()
            batch_loss += float(loss)
        
        print(f"Epoch: {n}, loss: {batch_loss:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Supervised Contrastive Learning')
    
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to the 2D numpy array of embeddings')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to the 1D numpy array of labels')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension in the network')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='Output layer dimension in the network')
    parser.add_argument('--drop_prob', type=float, default=0.4,
                        help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batches_per_epoch', type=int, default=750,
                        help='Number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=36,
                        help='Number of samples per batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin for the contrastive loss function')
    
    args = parser.parse_args()
    main(args)
