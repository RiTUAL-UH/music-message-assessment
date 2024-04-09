import pandas as pd
import glob
import os
import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

filename = "your_data"

tokenizer = AutoTokenizer.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
)

model = AutoModel.from_pretrained("j-hartmann/emotion-english-distilroberta-base")


# Sentences are encoded by calling model.encode()
def get_sentence_emb(whole_list_of_sentences):
    inputs = tokenizer(
        whole_list_of_sentences, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**inputs, return_dict=True)
    return_tensor = torch.tensor(outputs.pooler_output.detach().numpy())
    return return_tensor


all_data = pd.read_pickle(filename)

list_of_emo = []

for i in range(len(all_data["All_sent"])):
    emo_emb = get_sentence_emb(all_data["All_sent"][i])
    list_of_emo.append(emo_emb)
    if i % 10 == 0:
        print(i)

all_data["All_sent_emo_emb"] = list_of_emo

all_data.to_pickle("all_lyrics_emotion_emb.pkl")
print("Finished.")
