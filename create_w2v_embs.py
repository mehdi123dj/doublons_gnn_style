from transformers import BertTokenizer
import pandas as pd 
import os 
import gensim
import numpy as np 
import pickle 
import argparse
from gensim.models import Word2Vec 
import gensim.downloader


def generate_sentences(df,cat_cols):
    ret = ''
    df = df.fillna("")
    for c in cat_cols:
        ret+=df[c]+' '
    return ret 

def cat2vec(sentences,model,tokenizer):
    vocab = model.index2word
    feature_vector = np.zeros((len(sentences),model.vectors.shape[1]))
    vocab = model.index2word
    index = { vocab[i]: i for i in range(len(vocab))}
    for i in range(len(sentences)):
        s = sentences[i].lower()
        words =dict()
        tokens = tokenizer.tokenize(s)
        for w in tokens:
            try :
                words[w] =index[w]  
            except:
                pass
        v = np.zeros(model.vectors.shape[1])
        for w in words :
                v += model.vectors[words[w]]/len(words)

        feature_vector[i]+=v
    return feature_vector

def Embed_w2v(df,tokenizer,destination_dir):
    cat_cols = ['name', 'address', 'city','state', 'country','categories']
    daset =df[cat_cols]

    print('apply_w2v for cat2vec')
    c2v_model = gensim.downloader.load('glove-twitter-200')
    print('finished loading')
    sentences = list(generate_sentences(df,cat_cols))
    E = cat2vec(sentences,c2v_model,tokenizer)

    Emb_file = open(os.path.join(destination_dir,'W2V_Embeddings.pkl'),"wb")
    pickle.dump(E,Emb_file)

def main():

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data-dir",
        default='./data/raw',
        type=str,
    )

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    df = pd.read_csv(os.path.join(args.data_dir,'train.csv'))
    Embed_w2v(df,tokenizer,args.data_dir)

    
if __name__ == "__main__":
    main()
