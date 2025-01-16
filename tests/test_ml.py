import polars as pl
import duckdb
from pathlib import Path

#from szdetect import model as mod


features = pl.read_parquet(Path('./test_data/features.parquet'))

features_db = duckdb.sql('')