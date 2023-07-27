cracked_screen_identifier
==============================

This projects presents the complete end-to-end ML cycle of the cracked_screen_identifier app; an image classification project. Cracked_screen_identifier takes a phone image and predicts if the screen is broken or not. The simple web interface written with streamlitt and hosted with huggingface is available on the [Cracked_screen_identifier](https://huggingface.co/spaces/olumide/Cracked_Screen_Identifier).

This project was developed to ensure reproducibility.The project is structured below:

Project Organization
------------------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for model training.
    │   └── raw            <- The original, immutable data dump. This folder contains the readme to download the dataset
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports           
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    |   |   └── make_trainvaltest.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    |   |   └── dataloader.py
    │   │   └── model_architecture.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Usage
------------
To run the project, clone the repo and run the following commands:

1 . Clone the repo by running the following command in your terminal:

```bash
git clone https://github.com/olumideodetunde/ML_EndtoEnd_Projects.git
```
This will clone the entire repo to your local machine.

2. Install the requirements by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

3. Navigate to the data folder and select the raw folder. Download the dataset from the link in the readme file and unzip the dataset in the raw folder.



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
