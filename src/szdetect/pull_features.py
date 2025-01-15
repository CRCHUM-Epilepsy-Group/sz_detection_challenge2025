import duckdb
from pathlib import Path

FEATURE_GROUPS = {
    "efficiency": [
        "betweenness",
        "diversity_coef",
        "node_betweenness",
        "participation_coef",
        "module_degree_zscore",
        "eigenvector_centrality",
        "efficiency",
        "global_diffusion_efficiency",
        "global_rout_efficiency",
        "local_rout_efficiency",
    ],
    "connectivity": [
        "node_degree",
        "node_strength",
        "transitivity",
        "eigenvalues",
    ],
    "univariate": [
        "fuzzen",
        "linelength",
        "corr_dim",
        "band_power",
        "peak_alpha",
    ],
}


def pull_features(
    feature_dir: Path | str, label_file: Path | str, feature_group: str = "all"
):

    feature_dir = Path(feature_dir)
    feature_files = [str(f) for f in feature_dir.glob("*.parquet")]

    feature_rel = duckdb.read_parquet(feature_files)
    label_rel = duckdb.read_parquet(label_file)

    if feature_group == "all":
        feature_list = [
            feature
            for feature_list in FEATURE_GROUPS.values()
            for feature in feature_list
        ]
    else:
        feature_list = FEATURE_GROUPS[feature_group]

    query = """SELECT f.*, l.label
                FROM feature_rel f
                JOIN label_rel l
                    ON f.subject = l.subject 
                    AND f.session = l.session 
                    AND f.run = l.run 
                    AND f.timestamp = l.timestamp
                WHERE feature IN ? and l.training = TRUE
            """
    df = duckdb.execute(query, [feature_list]).pl()
    return df
