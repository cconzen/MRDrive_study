# MRDrive Study Data Processing & Visualisation

This repository contains scripts developed for the study conducted as part of my Master's thesis. The code is used to clean, transform, analyse, and visualise data collected during the experiment.

## Requirements

The language of the scripts is Python / Jupyter Notebooks.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
.
├── DATA/                               # Raw and processed datasets (not included)
├── preprocess
|     ├── preprocess_score_data.ipynb   # preprocess raw questionnaires into clean dataframes
|     └── trim_vids_and_logs.py         # trim recorded videos and logs to when the start button is pressed
├── visualise/
|     ├── linearmodels.ipynb            # linear mixed model scripts
|     ├── lmm_blocks.ipynb              # linear mixed models and other plots
|     ├── vis_demographics.ipynb        # spearman matrix, general demographics plots
|     └── vis_scores.ipynb              # DV boxplots, tests
├── requirements.txt
└── README.md
```

## Data

The experimental data used in this project are **not included** in this repository due to privacy and ethical considerations.

If you wish to reproduce the analyses, you will need access to the original study data.
