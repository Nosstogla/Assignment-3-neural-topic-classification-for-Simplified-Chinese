from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Classifier

if __name__ == "__main__":
    argparser = ArgumentParser(
        prog="Simplified chinese pytorch train model",
        description="Train PyTorch model on sentence embeddings",)

    argparser.add_argument("--data", help="File to process", required = True)
    argparser.add_argument("--epochs", help="Number of training epochs (default 10)", type = int, default = 10)
    argparser.add_argument("--batchsize", help="Size of each training batch (default 32)", type = int, default = 32)
    argparser.add_argument("--output",help ="File to save the trained PyTorch model", required=True)
    args = argparser.parse_args()

    print("Working...")
    data = np.load(args.data)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))

    model = Classifier(input_dim, 128, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        perm = torch.randperm(X.size(0))

        total_loss = 0

        for i in range(0, X.size(0), args.batchsize):
            indices = perm[i:i+args.batchsize]
            batch_x = X[indices]
            batch_y = y[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved as {args.output}")