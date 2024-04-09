import pandas as pd
import glob
import os
from sentence_transformers import SentenceTransformer
import torch

# assume the pickled dataframe file has a lyrics column ['All_sent']
file_name = "./your_lyrics_data.pkl"

model = SentenceTransformer("stsb-bert-base", device="cuda:0")


# Sentences are encoded by calling model.encode()
def get_sentence_emb(whole_list_of_sentences):
    embeddings = model.encode(whole_list_of_sentences, convert_to_tensor=True)
    return embeddings


print("Working on: ", file_name)
all_data = pd.read_pickle(file_name)
all_data["All_sent_emb"] = all_data["All_sent"].apply(get_sentence_emb)
all_data.to_pickle(file_name[:-4] + "_emb.pkl")
print(file_name[:-4] + "_emb.pkl -- finished.")
