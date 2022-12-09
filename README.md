# deepfocus

this project is for the in-focus pixels segmentation.

using this model, we could segment the in-focus pixels in different focal planes. This enbales the virtual optical sectioning for wildefield microscopy.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- how to configure the repo
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- the data as the final input to models
    │   ├── processed      <- data after preprocessing
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- adopted models
    ├── models_weight      <- trained weights
    │
    ├── notebooks          <- Jupyter notebooks for data inspection
    │
    ├── references         <- literature, papers and models
    │
    ├── reports            <- part of the results
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


![img][[reports/featureMap/autofocusing.svg](https://github.com/casus/deepfocus/blob/96590a07c8f2b454b8d1a107e632efec68b3749a/reports/featureMap/autofocusing.svg)](https://github.com/casus/deepfocus/blob/653c83b43a056cf764005375913cc5e7f852a2e9/reports/UNet2D_vanilla/fig1.png)
