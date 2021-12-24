import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":

    data_path = sys.argv[1]

    with open(os.path.join(data_path, 'entities.dict')) as fin:
        ent2idx = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            ent2idx[entity] = int(eid)

    df = pd.read_csv(f'{data_path}/numerical_literals.txt', header=None, sep='\t')

    rel2idx = {v: k for k, v in enumerate(df[1].unique())}

    num_lit = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)

    for i, (s, p, lit) in enumerate(df.values):
        try:
            num_lit[ent2idx[s.lower()], rel2idx[p]] = lit
        except KeyError:
            continue

    np.save(f'{data_path}/numerical_literals.npy', num_lit)
