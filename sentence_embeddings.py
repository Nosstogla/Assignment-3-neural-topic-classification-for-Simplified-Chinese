from argparse import ArgumentParser
import pandas as pd
from gensim.models import FastText
import numpy as np

def sentence_to_vec(sentence, model):
    vectors = []
    for char in sentence:
        if char in model.wv:
            vectors.append(model.wv[char])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis = 0)


if __name__ == "__main__":
    argparser = ArgumentParser(
        prog="Simplified chinese sentence embeddings",
        description="Create sentence embeddings from TSV files",)

    argparser.add_argument("--input", help="File to process", required = True)
    argparser.add_argument("--model", help="FastText model file", required = True)
    argparser.add_argument("--output", help ="File to save the sentence embeddings", required=True) 
    args = argparser.parse_args()

    print("Working...")
    df = pd.read_csv(args.input, sep= "\t")
    model = FastText.load(args.model)

    embeddings = []
    
    for text in df["text"].astype(str):
        vec = sentence_to_vec(text, model)
        embeddings.append(vec)

    embeddings = np.array(embeddings)

    df["label"] = df["category"].astype("category").cat.codes

    np.savez(args.output, X = embeddings, y = df["label"].values)
    print(f"Saved embeddings to {args.output}")