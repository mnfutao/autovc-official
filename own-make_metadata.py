"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
from datasets.other_speech.speaker import speaker_list, speaker2idx
from tqdm import tqdm


num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDirPath = './datasets/other_speech/'
melsDirName = 'mels_4'
melsDirPath = os.path.join(rootDirPath, melsDirName)
speakerEmbeddingPath = os.path.join(rootDirPath, 'new_char_emb.npy')

mels_list = os.listdir(melsDirPath)
speakerEmbeddin = np.load(speakerEmbeddingPath)

results = {}
for i in tqdm(range(len(mels_list))):
    c_speaker_name = mels_list[i].strip().split('-')[0]
    if c_speaker_name not in results:
        c_speaker_id = speaker2idx[c_speaker_name]
        c_speaker_embedding = speakerEmbeddin[c_speaker_id]
        results[c_speaker_name] = [c_speaker_id, c_speaker_embedding]
    c_mel_path = os.path.join(melsDirName, mels_list[i].strip())
    results[c_speaker_name].append(c_mel_path)

final_metadata = []
for key in results:
    final_metadata.append(results[key])

with open(os.path.join(rootDirPath, 'train.pkl'), 'wb') as handle:
    pickle.dump(final_metadata, handle)


