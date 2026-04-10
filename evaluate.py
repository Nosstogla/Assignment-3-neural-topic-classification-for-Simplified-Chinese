from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from model import Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    argparser = ArgumentParser(
        prog="Simplified chinese evaluation",
        description="Evaluate a trained PyTorch classifier on sentence embeddings",)

    argparser.add_argument("--data", help="File to process", required = True)
    argparser.add_argument("--model", help="PyTorch model file", required = True)
    args = argparser.parse_args()

    print("Working...")
    data = np.load(args.data)
   
    X = torch.tensor(data["X"], dtype=torch.float32)
    y_test = data["y"]

    input_dim = X.shape[1]
    num_classes = len(set(y_test))

    model = Classifier(input_dim, 128, num_classes)
    model.load_state_dict(torch.load(args.model, weights_only = True))
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=1).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    df_train = pd.read_csv("Data/train.tsv", sep="\t")
    categories = list(df_train["category"].astype("category").cat.categories)
    
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(categories)))
    cm_df = pd.DataFrame(cm, index=categories, columns=categories)

    print("\nEvaluation Results:")
    print(f"Accuracy:  {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1-score:  {f1:.5f}\n")
    
    print("Confusion Matrix:")
    print(cm_df.to_string())
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.title("Confusion Matrix")
    plt.figtext(0.5, -0.1,
                f"Accuracy: {accuracy:.5f} | Precision: {precision:.5f} | Recall: {recall:.5f} | F1: {f1:.5f}",
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("\nHeatmap also saved as confusion_matrix.png")

