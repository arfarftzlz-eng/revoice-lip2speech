import os

import torch


UNIGRAM1000_PATH = os.path.join(
    os.path.dirname(__file__), "labels", "unigram1000_units.txt"
)
UNIGRAM1000_LIST = (
    ['<blank>']
    + [
        _.split()[0]
        for _ in open(UNIGRAM1000_PATH, encoding="utf-8").read().splitlines()
    ]
    + ['<eos>']
)


def ids_to_str(token_ids, char_list):
    tokenid_as_list = list(map(int, token_ids))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    return "".join(token_as_list).replace("<space>", " ")


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
