import numpy as np
import polars as pl
from pathlib import Path
from szdetect.model import grouper

# Example
labels_file = Path('./test_data/labels.parquet')
labels = pl.read_parquet(labels_file)
test_rec_sz = labels.filter(pl.col('unique_id')=='siena_bids_01_01_00')
annotations = np.array(test_rec_sz['label']) 
true_indexes = np.array([i for i, x in enumerate(annotations) if x == True])


true_events = dict(enumerate(grouper(true_indexes, thres=1), 1))

for i in true_events:
    print('sz onset', test_rec_sz[int(true_events[i][0]), 'timestamp'])