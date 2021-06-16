import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
import argparse
from pathlib import Path
from typing import List, Tuple
from datasets.other_speech.speaker import speaker2idx, speaker_list
from torch.nn.functional import pad
from torch import Tensor


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def pad_seq(x: Tensor, base: int = 32) -> Tuple[Tensor, int]:
    len_out = int(base * ceil(float(len(x)) / base))
    len_pad = len_out - len(x)
    assert len_pad >= 0
    return pad(x, (0, 0, 0, len_pad), "constant", 0), len_pad

def main(
    model_path: Path,
    speaker_embedding_path: Path,
    src_speaker: str,
    tar_speaker: str,
    src_mel_path: Path,
    result_path: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = Generator(16, 256, 512, 16).eval().to(device)
    # G = Generator(32,256,512,32).eval().to(device)

    checkpoint = torch.load(model_path)
    model_dict = checkpoint['state_dict']
    G.load_state_dict(model_dict)

    model = G

    speaker_emb = torch.from_numpy(np.load(str(speaker_embedding_path)).astype(np.float32)).to(device)

    # src_mel
    src_mel = torch.from_numpy(np.load(str(src_mel_path))).to(device)

    # src_embed
    src_speaker_id = speaker2idx[src_speaker]
    src_emb = speaker_emb[src_speaker_id].unsqueeze(0)

    # tar_embed
    tar_speaker_id = speaker2idx[tar_speaker]
    tar_emb = speaker_emb[tar_speaker_id].unsqueeze(0)

    src_mel, len_pad = pad_seq(src_mel)
    src_mel = src_mel[None, :]

    with torch.no_grad():
        _, mel, _ = model(src_mel, src_emb, tar_emb)

    mel = mel.squeeze(0)

    mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]

    final_result_path = str(result_path / src_mel_path.name.split('/')[-1].split(".")[0]) + '_vc'
    np.save(final_result_path, mel.cpu().numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, default=Path('./logdir/model.pth'))
    parser.add_argument("speaker_embedding_path", type=Path, default=Path('datasets/other_speech/new_char_emb.npy'))
    parser.add_argument("src_speaker", type=str, default='yaya')
    parser.add_argument("tar_speaker", type=str, default='biaobei')
    parser.add_argument("src_mel_path", type=Path, default=Path('./spmel/p225/p225_023.npy'))
    parser.add_argument("result_path", type=Path, default=Path('./logdir'))
    main(**vars(parser.parse_args()))
