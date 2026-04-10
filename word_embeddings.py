from argparse import ArgumentParser
import pandas as pd
from gensim.models import FastText


if __name__ == "__main__":
    argparser = ArgumentParser(
        prog="Simplified chinese word embeddings",
        description="Train FastText embeddings from TSV files",) 

    argparser.add_argument("--filenames", required=True, help="TSV files to train embeddings on", nargs="+")
    argparser.add_argument("--dimensionsize", help="The dimensionality of word embeddings, default is 300", type=int, default = 300)
    argparser.add_argument("--output", required=True, help="File to save the trained FastText model")
    args = argparser.parse_args()

    print("Working...")
    all_texts = []
    for filename in args.filenames:
        df = pd.read_csv(filename, sep="\t")
        all_texts.extend(df["text"].astype(str).tolist())

    tokenized = [list(text) for text in all_texts]
    model = FastText(vector_size=args.dimensionsize, min_count=1, window=5 ) 
    
    model.build_vocab(tokenized)
    model.train(tokenized, total_examples=len(tokenized), epochs=20)
    
    model.save(args.output)
    print(f"FastText model saved to {args.output}")