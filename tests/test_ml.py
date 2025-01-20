import polars as pl
#import duckdb
from pathlib import Path

#from szdetect import model as mod

features_file = Path('./test_data/features/tuh_sz_bids_007_00_00.parquet')
features = pl.read_parquet(features_file)

