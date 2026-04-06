# Benchmark Results and Experimental Setup

Below are the key tables summarizing the datasets we used and the link‑prediction results obtained with various methods, followed by detailed information about our software setup, installation steps, datasets, and how to reproduce our experiments.

---

## Dataset Statistics

| Type          | Dataset          | # Nodes | # Edges | # Features | # Classes |
| ------------- | ---------------- | ------: | ------: | ---------: | --------: |
| Homophilous   | Cora             |   2,708 |  10,556 |      1,433 |         7 |
| Homophilous   | CiteSeer         |   3,327 |   9,104 |      3,703 |         6 |
| Homophilous   | PubMed           |  19,717 |  88,648 |        500 |         3 |
| Homophilous   | FB Page-Page     |  22,470 | 171,002 |         31 |         4 |
| Homophilous   | Coauthor-Physics |  34,493 | 495,924 |      8,415 |         5 |
| Homophilous   | Facebook         |   4,039 |  88,234 |      1,283 |       193 |
| Homophilous   | DBLP             |  17,716 | 105,734 |      1,639 |         4 |
| Heterophilous | Roman-Empire     |  22,662 |  32,927 |        300 |        18 |
| Heterophilous | Amazon-Ratings   |  24,492 |  93,050 |        300 |         5 |
| Heterophilous | Questions        |  48,921 | 153,540 |        301 |         2 |
| Heterophilous | Chameleon        |   2,277 |  36,101 |      2,325 |         5 |
| Heterophilous | Actor            |   7,600 |  33,544 |        931 |         5 |
| Heterophilous | Crocodile        |   1,118 |  15,620 |      1,438 |         3 |

---

## Link‑Prediction Results on Homophilous Graphs

| Method               | Cora (HR\@100) | Citeseer (HR\@100) | Pubmed (HR\@100) | FB Page‑Page (MRR) | Facebook (HR\@100) | Coauthor‑Physics (MRR) | DBLP (HR\@10) |
| -------------------- | -----------: | ---------------: | -------------: | ---------------: | ---------------: | -------------------: | ----------: |
| CN                   |        33.92 |            29.79 |          23.13 |            17.85 |            84.38 |                18.57 |        32.8 |
| AA                   |        39.85 |            35.19 |          27.38 |            22.60 |            88.14 |                22.31 |       21.13 |
| RA                   |        41.07 |            33.56 |          27.03 |            20.54 |            92.58 |                21.46 |       22.47 |
| GCN                  |   66.79±1.65 |       67.08±2.94 |     53.02±1.39 |        11.26±1.6 |       92.85±0.61 |           14.68±3.40 |  33.30±4.74 |
| SAGE                 |   55.02±4.03 |       57.01±3.57 |     44.29±1.44 |       10.44±2.48 |        68.50±8.6 |           13.07±1.02 |  31.06±5.98 |
| GAE                  |   89.01±1.32 |       91.78±0.94 |     78.81±1.64 |       12.93±0.66 |       92.68±2.58 |           15.83±1.67 |  41.38±3.72 |
| Neo‑GNN              |   80.42±1.34 |       84.67±1.42 |     73.93±1.19 |       12.43±0.22 |       91.24±0.77 |           20.94±3.94 |  50.05±3.40 |
| BUDDY                |   88.00±0.44 |       92.93±0.27 |     74.10±0.78 |       16.94±1.37 |       87.56±1.43 |           14.26±1.82 |  31.74±6.09 |
| NCN                  |   89.05±0.96 |       91.56±1.43 |     79.05±1.16 |        9.16±1.96 |       93.67±0.82 |           29.05±3.48 |  51.26±3.26 |
| NCNC                 |   89.65±1.36 |       93.47±0.95 |     81.29±0.85 |       14.03±7.88 |       92.78±2.00 |           20.99±5.09 |  42.82±4.12 |
| NCN + Class label    |   95.71±1.10 |       96.96±0.37 |     90.81±1.13 |       11.27±4.62 |       93.69±0.62 |           27.04±3.93 |  51.75±2.55 |
| CGLE(NCN) (True CL)  |   95.77±0.62 |       97.27±0.74 |     90.49±0.54 |       12.06±5.57 |       93.75±0.79 |           26.97±4.32 |  51.33±2.00 |
| NCNC + Class label   |   88.63±1.72 |       92.46±1.05 |     82.02±1.51 |       12.72±8.41 |       92.95±0.62 |           21.48±6.47 |  42.54±4.28 |
| CGLE(NCNC) (True CL) |   91.41±1.36 |       92.31±0.14 |     82.06±0.13 |       23.84±6.15 |       93.92±0.56 |           21.24±3.06 |  49.00±3.10 |
| CGLE(NCN)‑kmeans     |   94.27±0.94 |       95.89±1.84 |     90.44±0.83 |        7.84±1.28 |       93.99±0.59 |           27.29±3.47 |  52.86±1.48 |
| CGLE(NCNC)‑kmeans    |   94.80±0.96 |       96.90±1.12 |     91.65±0.60 |       16.32±5.70 |       93.61±0.90 |           24.94±4.42 |  48.88±3.21 |

---

## Link‑Prediction Results on Heterophilous Graphs

| Method               | Roman Empire (MRR) | Amazon‑ratings (MRR) | Questions (HR\@100) |  Chameleon (MRR) | Actor (HR\@100) |
| -------------------- | ---------------: | -----------------: | ----------------: | -------------: | ------------: |
| NCN                  |       54.29±0.86 |         55.90±7.51 |        62.25±1.75 |     76.79±1.33 |    53.18±1.65 |
| NCNC                 |      28.23±12.51 |         72.63±6.69 |        62.93±1.73 |     74.75±8.37 |    50.77±3.07 |
| NCN + Class          |       52.32±1.96 |         59.88±8.72 |        63.89±1.40 |     77.09±2.92 |    51.01±2.35 |
| NCNC + Class         |      32.35±11.88 |         67.56±3.17 |        63.89±1.40 |     73.68±7.78 |    51.48±1.19 |
| CGLE(NCN) (True CL)  |       54.01±0.71 |         64.68±8.25 |        63.02±1.55 | **81.15±3.09** |    53.37±1.71 |
| CGLE(NCNC) (True CL) |       52.23±2.31 |         70.62±5.96 |        63.44±1.57 |     77.88±8.29 |    51.07±4.31 |
| CGLE‑kmeans(NCN)     |       53.19±1.44 |         64.03±6.87 |        61.33±2.98 |     77.32±4.19 |    54.82±1.57 |
| CGLE‑kmeans(NCNC)    |       53.82±2.57 |         73.67±5.11 |        63.95±2.82 |     77.87±5.45 |    51.42±3.87 |

---

| Metric  | Dataset          |        NCNC |            k=2 |            k=5 |       k=10 |           k=15 |           k=20 |
| ------- | ---------------- | ----------: | -------------: | -------------: | ---------: | -------------: | -------------: |
| HR\@100 | Cora             |  89.65±1.36 | **94.80±0.96** |     94.54±0.78 | 94.58±0.95 |     94.39±1.36 |     94.31±1.35 |
| HR\@100 | Citeseer         |  93.47±0.95 |     96.55±1.65 |     96.66±1.52 | 96.22±2.49 |     96.30±2.44 | **96.90±1.12** |
| HR\@100 | Pubmed           |  81.29±0.85 |     91.52±0.37 |     91.30±0.70 | 91.36±0.62 |     91.23±0.26 | **91.65±0.60** |
| HR\@100 | Facebook         |  92.78±2.00 | **93.61±0.90** |     93.36±1.74 | 93.44±1.15 |     93.38±1.78 |     93.54±1.43 |
| MRR     | FB Page‑Page     |  14.03±7.88 |     12.97±4.48 | **16.32±5.70** | 11.58±3.51 |     13.07±2.65 |     15.32±5.22 |
| MRR     | Coauthor‑Physics |  20.99±5.09 |     23.81±2.31 | **24.94±4.42** | 24.28±2.51 |     23.67±3.26 |     22.47±1.43 |
| HR\@10  | DBLP             |  42.82±4.12 | **48.88±3.21** |     49.14±2.99 | 49.08±4.50 |     48.11±3.99 |     46.96±5.56 |
| MRR     | Roman Empire     | 28.23±12.51 |     52.93±1.95 | **53.82±2.57** | 53.50±2.19 |     53.25±2.48 |     53.35±1.74 |
| MRR     | Amazon‑ratings   |  72.73±6.69 | **73.67±5.11** |     69.96±6.86 | 72.74±4.56 |     69.23±8.47 |     68.50±7.54 |
| MRR     | Chameleon        |  74.75±8.37 |    77.61±11.03 |     77.11±6.37 | 77.28±7.44 | **77.87±5.45** |     75.97±8.60 |
| HR\@100 | Questions        |  62.93±1.73 |     63.00±2.43 |     63.21±2.79 | 63.59±2.40 |     63.15±2.53 | **63.95±2.82** |
| HR\@100 | Actor            |  50.77±3.07 |     51.15±3.65 | **51.42±3.87** | 51.39±3.39 |     51.72±2.62 |     51.09±3.69 |

---

| Metric  | Dataset          |        NCN |            k=2 |            k=5 |           k=10 |           k=15 |           k=20 |
| ------- | ---------------- | ---------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| HR\@100 | Cora             | 89.05±0.96 |     94.18±1.00 |     94.23±0.92 |     94.21±0.94 | **94.27±0.94** |     94.16±0.90 |
| HR\@100 | Citeseer         | 91.56±1.43 |     95.45±2.71 |     95.80±2.22 |     95.56±2.22 |     95.67±2.79 | **95.89±1.84** |
| HR\@100 | Pubmed           | 79.05±1.16 |     90.04±0.78 |     90.39±0.81 |     90.38±0.86 | **90.44±0.83** |     90.38±0.82 |
| HR\@100 | Facebook         | 93.67±0.82 |     93.59±0.77 |     93.85±0.50 |     93.74±0.67 | **93.99±0.59** |     93.55±0.63 |
| MRR     | FB Page‑Page     |  9.16±1.96 |      7.60±2.03 |      7.37±1.84 |      7.26±1.24 |      7.76±1.21 |  **7.84±1.28** |
| MRR     | Coauthor‑Physics | 29.05±3.48 |     26.08±3.19 |     24.91±3.45 |     26.89±2.68 | **27.29±3.47** |     26.48±0.32 |
| HR\@10  | DBLP             | 51.26±3.26 |     51.53±2.17 | **52.86±1.48** |     50.99±2.84 |     51.49±2.10 |     51.11±2.30 |
| MRR     | Roman Empire     | 54.29±0.86 | **53.19±1.44** |     52.46±1.85 |     52.25±1.81 |     52.54±1.74 |     52.98±2.25 |
| MRR     | Amazon‑ratings   | 55.90±7.51 |     61.55±5.46 |     60.93±4.35 |     61.92±6.52 |     60.38±7.23 | **64.03±6.87** |
| MRR     | Chameleon        | 76.79±1.33 |     76.55±3.53 |     75.24±7.43 |     75.47±5.23 | **77.32±4.19** |     75.63±6.22 |
| HR\@100 | Questions        | 62.25±1.75 |     60.57±3.42 |     61.33±2.98 | **61.83±1.03** |     59.70±3.39 |     60.88±2.46 |
| HR\@100 | Actor            | 53.18±1.65 | **54.82±1.57** |     54.56±2.48 |     54.64±1.85 |     54.32±1.91 |     54.72±1.72 |

---

## Latest Setup

* **Python**: 3.10.14
* **PyTorch**: 2.3.1 (CUDA 12.1)
* **PyTorch Geometric**: 2.5.3
* **OGB**: 1.3.6

**Hardware:**
Experiments were conducted using NVIDIA A100 (80GB) and NVIDIA V100 GPUs.

---

## Installation Guide

1. Install [PyTorch](https://pytorch.org/).
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. Install [OGB](https://ogb.stanford.edu/docs/home/).

---

## Datasets

1. **Facebook Page-Page Network**:
   We used the dataset from [Facebook Large Page-Page Network](https://snap.stanford.edu/data/facebook-large-page-page-network.html).
   Since not all features have the maximum dimension of 31, we concatenated them with zeros to make all node feature dimensions consistent at 31.
   The dataset can be found in the `CGLE/datasets/fb_page` directory.

2. **Other Datasets**:
   We used datasets from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.6.0/modules/datasets.html).

---

## Getting Started

Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd CGLE
```

### Running Experiments

* To reproduce **NCNC**:

  ```bash
  bash ncnc_run.sh
  ```

* To reproduce **NCN**:

  ```bash
  bash ncn_run.sh
  ```

* To reproduce **NCNC ⊕ Class labels**:

  ```bash
  bash ncnc_concat_y.sh
  ```

* To reproduce **NCN ⊕ Class labels**:

  ```bash
  bash ncn_concat_y.sh
  ```

* To reproduce **CGLE(NCNC)**:

  ```bash
  bash cgle_ncnc.sh
  ```

* To reproduce **CGLE(NCN)**:

  ```bash
  bash cgle_ncn.sh
  ```

* To reproduce **CGLE(NCNC) with k-means**:

  ```bash
  bash cgle_ncnc_k-means.sh
  ```

* To reproduce **CGLE(NCN) with k-means**:

  ```bash
  bash cgle_ncn_k-means.sh
  ```

---

## Acknowledgments

This implementation was inspired by the following repositories:

* [HeaRT](https://github.com/Juanhui28/HeaRT.git)
* [Neural Common Neighbor](https://github.com/GraphPKU/NeuralCommonNeighbor.git)
