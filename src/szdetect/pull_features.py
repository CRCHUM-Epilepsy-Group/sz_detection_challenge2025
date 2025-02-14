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
    test_only: bool = False,
    train_only: bool = True,
    inference: bool = False,
    num_eegs: int | None = None,
    step_size: int = 4,
    start_eeg: int | None = None,
    end_eeg: int | None = None,
):
    """
        Extracts and filters features from Parquet files and joins them with labels.

    This function reads a directory of Parquet files containing features and optionally joins them with labels.
    It filters results based on a feature group and provides options for training/testing data selection.

    Parameters:
    ----------
    feature_dir : Path | str
        Path to the directory containing feature Parquet files.
    label_file : Path | str
        Path to the Parquet file containing labels.
    feature_group : str, optional
        Feature group to filter. Defaults to "all". Options:
        - "all": All features in FEATURE_GROUPS
        - "efficiency": Network efficiency features
        - "connectivity": Network connectivity features
        - "univariate": Signal analysis features
    test_only : bool, optional
        Filter for test data only. Defaults to False.
    train_only : bool, optional
        Filter for training data only. Defaults to True.
    inference : bool, optional
        Run in inference mode without labels. Defaults to False.
    num_eegs : int | None, optional
        Limit number of EEGs to process. Defaults to None.
    step_size : int, optional
        Step size for epoch conversion. Defaults to 4.
    start_eeg : int | None, optional
        Starting index for EEG selection. Defaults to None.
    end_eeg : int | None, optional
        Ending index for EEG selection. Defaults to None.

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

    if start_eeg is not None and end_eeg is not None:
        end_eeg = min(end_eeg, len(feature_files))
        feature_files = feature_files[start_eeg:end_eeg]

    feature_rel = duckdb.read_parquet(feature_files)  # type: ignore

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
        if train_only:
            join_clause = " AND l.training = TRUE"
        elif test_only:
            join_clause = " AND l.training = FALSE"
        else:
            join_clause = ""
        join_where_clause = where_clause + join_clause

    query = f"""SELECT
                    f.dataset_name,
                    f.region_side,
                    f.subject,
                    f.session,
                    f.run,
                    f.unique_id,
                    f.timestamp,
                    CAST(f.epoch AS INTEGER) * {step_size} AS second,
                    f.feature,
                    f.freqs,
                    AVG(f.value) AS value
                    {", l.label" if not inference else ""}
                FROM feature_rel AS f
                {join_where_clause}
                GROUP BY
                    f.dataset_name,
                    f.subject, 
                    f.session, 
                    f.run, 
                    f.unique_id,
                    f.timestamp,
                    f.second,
                    f.feature, 
                    f.freqs,
                    f.region_side,
                    f.epoch
                    {", l.label" if not inference else ""}
            """
    df = duckdb.execute(query, [feature_list]).pl()

    return df
