digitalConfocal
==============================

digital confocal microscopy

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

the pipeline of the proejct works as below:
![img](https://github.com/casus/deepfocus/blob/672ed2a76b50361f56e411a9e3e543c0bb11f82b/reports/UNet2D_vanilla/fig1.png)

part of the results by segmentation are listed below:
![img](https://github.com/casus/deepfocus/blob/672ed2a76b50361f56e411a9e3e543c0bb11f82b/reports/UNet2D_vanilla/fig4.png)

when applied on the stack of the images, this model enables the virtual optical sectioning on widefiled microscopy. The segmented results reveal the 3D information of the targets.

<div align=center>
<img src="https://github.com/casus/deepfocus/blob/672ed2a76b50361f56e411a9e3e543c0bb11f82b/reports/UNet2D_vanilla/test2.gif">
</div>

