import duckdb
import random
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
    feature_dir: Path | str,
    label_file: Path | str,
    feature_group: str = "all",
    train_only: bool = True,
    inference: bool = False,
    num_eegs: int | None = None,
):
    """
    Extracts and filters features from Parquet files and joins them with labels.

    This function reads a directory of Parquet files containing features and a single Parquet file containing labels.
    It joins the features and labels on specified keys and filters the results based on a feature group.
    By default, it filters to include only rows where the labels indicate training data.

    Parameters:
    ----------
    feature_dir : Path | str
        Path to the directory containing feature Parquet files.
    label_file : Path | str
        Path to the Parquet file containing labels.
    feature_group : str, optional
        The name of the feature group to filter. Defaults to "all",
        which includes all features in the `FEATURE_GROUPS` dictionary.
        Possible values:
        - "all": Includes all features in the `FEATURE_GROUPS`.
        - "efficiency": Includes features related to network efficiency:
            - "betweenness", "diversity_coef", "node_betweenness", "participation_coef",
              "module_degree_zscore", "eigenvector_centrality", "efficiency",
              "global_diffusion_efficiency", "global_rout_efficiency", "local_rout_efficiency".
        - "connectivity": Includes features related to network connectivity:
            - "node_degree", "node_strength", "transitivity", "eigenvalues".
        - "univariate": Includes features related to signal analysis:
            - "fuzzen", "linelength", "corr_dim", "band_power", "peak_alpha".
    train_only : bool, optional
        Whether to filter for rows where the label file indicates `training = TRUE`. Defaults to True.

    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame containing the joined and filtered features and labels.

    Example:
    --------
    >>> df = pull_features(
    ...     feature_dir="/path/to/features",
    ...     label_file="/path/to/labels.parquet",
    ...     feature_group="efficiency",
    ...     train_only=True
    ... )
    >>> print(df)
    """
    feature_dir = Path(feature_dir)
    feature_files = [str(f) for f in feature_dir.glob("*.parquet")]
    if num_eegs is not None:
        random.shuffle(feature_files)
        feature_files = feature_files[:num_eegs]

    feature_rel = duckdb.read_parquet(feature_files)  # type: ignore

    # TODO: add parameter to avoid loading features (inference = True)
    if inference:
        label_rel = None
    else:
        label_rel = duckdb.read_parquet(str(label_file))

    if feature_group == "all":
        feature_list = [
            feature
            for feature_list in FEATURE_GROUPS.values()
            for feature in feature_list
        ]
    else:
        feature_list = FEATURE_GROUPS[feature_group]

    if inference:
        join_where_clause = "WHERE feature IN ?"
    else:
        where_clause = """
            JOIN label_rel AS l
                ON f.subject = l.subject 
                AND f.session = l.session 
                AND f.run = l.run 
                AND f.timestamp = l.timestamp
            WHERE feature IN ?"""
        join_clause = " AND l.training = TRUE" if train_only else ""
        join_where_clause = where_clause + join_clause

    # TODO: average over brain region if brain_region is not None
    query = f"""SELECT
                    f.dataset_name,
                    f.region_side,
                    f.subject,
                    f.session,
                    f.run,
                    f.timestamp,
                    f.feature,
                    f.freqs,
                    AVG(f.value) AS value
                    {", l.unique_id, l.label" if not inference else ""}
                FROM feature_rel AS f
                {join_where_clause}
                GROUP BY
                    f.dataset_name,
                    f.subject, 
                    f.session, 
                    f.run, 
                    f.timestamp, 
                    f.feature, 
                    f.freqs, 
                    f.region_side
                    {", l.unique_id, l.label" if not inference else ""}
            """
    df = duckdb.execute(query, [feature_list]).pl()

    return df
