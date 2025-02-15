# An Efficient Pipeline for Seizure Detection on Scalp EEG Combining Multi-scale Features and Gradient Boosting

Submission for the 2025 AI in Epilepsy Conference [Seizure Detection Challenge](https://epilepsybenchmarks.com/challenge/) from CRCHUM's Epilepsy Research Team, Montreal, Canada

## Authors (alphabetic order)

Elie Bou Assi, Daniel Galindo, Oumayma Gharbi, François Hardy, Amirhossein Jahani, Émile Lemoine, Isabel Sarzo Wabi

## Docker Images

The pretrained model can be found here: [Link to Docker images](https://github.com/orgs/CRCHUM-Epilepsy-Group/packages/container/package/sz_detect_crchum).

### Instructions

#### Prerequisites
- Docker installed
- Optimized for at least 50GB RAM and 10 CPU cores
- EEG file in EDF format (BIDS-compliant naming)

#### Installation

``` sh
docker pull ghcr.io/crchum-epilepsy-group/sz_detect_crchum:latest
```
#### Usage
1. Run detection on a single file:

``` sh
docker run --rm \
  -v /path/to/your/eeg:/data \
  -e INPUT=/data/your_eeg.edf \
  -e OUTPUT=/data/results.tsv \
  ghcr.io/crchum-epilepsy-group/sz_detect_crchum:the-og
```

2. Expected output:
- Results will be saved in TSV format ([HED-score compliant](https://hed-schemas.readthedocs.io/en/latest/hed_score_schema.html))
- Processing time: ~5 minutes per hour of EEG

#### Troubleshooting
- Ensure EDF file follows BIDS naming convention
- Check read/write permissions in mounted volumes
- For memory issues, increase Docker's resource allocation


## Abstract

**Rationale and Algorithm**: Seizure detection on scalp EEG faces challenges due to poor signal-to-noise ratio and heterogeneity in seizure patterns and localizations. We developed a feature extraction algorithm that captures seizure-related changes across various timescales and frequencies. Our approach combines linear, non-linear, and connectivity features, and uses these as input into a Gradient Boosted Trees model regularized by a  post-processing algorithm.

**Data Processing**: EEGs were segmented into overlapping 10s windows (4s overlap) using all 19 standard EEG channels in average referential montage. We applied the Continuous Wavelet Transform with Morlet wavelets to decompose the signal into 8 frequency bands (from 3 Hz to 50 Hz, one 3–40 Hz band). After frequency-dependent downsampling, we extracted linear (band power), non-linear (fuzzy entropy and line length), and connectivity features (including betweenness, efficiency, eigenvector centrality, node strength) from the filtered signals. The scaled features were then processed through a machine learning pipeline combining minimum Redundancy-Maximum Relevance feature selection and XGBoost classification. The epoch-based predictions were regularized using the Firing Power algorithm, which consists in applying a moving average across $\tau$ segments and binarizing the results with a fixed threshold $T$.

**Training and Validation**: The model was trained on a subset of the Temple University Hospital and the Siena Hospital Seizure Detection datasets. We conducted an initial feature exploration step through visual exploration of 1,507 EEGs (training set) to narrow down the set of features and frequency bands. Hyperparameter selection was done with a random search over a nested cross-validation (100 iterations, 5-fold inner loop, 3-fold outer loop). The criterion for the cross-validation was the epoch-wise F1-score. The outer-loop predictions were used to select the optimal Firing Power’s $\tau$ and $T$ values.

**Performance**: We tested the best model on a held-out set of 453 EEGs from the same datasets, without overlap between subjects. With a $\tau = 12$ and $T = 0.4$, the model achieved an average event-based F1 score of 0.72. For the final submission, we further optimized $\tau$ and $T$ on this held-out set.

**Complexity**: Our optimized implementation employs frequency-domain convolutions and channel-wise parallelization, achieving logarithmic-linear complexity with signal length. Processing time averages 5 minutes per hour of EEG using 10 CPU cores and 50 GB RAM.
