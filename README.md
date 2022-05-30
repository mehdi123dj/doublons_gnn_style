# Kaggle Competition-Foursquare

Notebook for the kaggle competition  : https://www.kaggle.com/competitions/foursquare-location-matching

## File tree structure

    ├── data                      
    │   ├── raw                   # Folder where we can find the raw files to be taken to create the graph
    │   ├── processed             # Folder for the processed dataset to be used for training our model
    ├── models                    # Folder for stored model
    ├── pytorch-geometric         # Folder for source files
    ├── LICENSE
    └── README.md

## Embedding for text
Using a language agnostic bert based model to transform a concatenation of the name and categories of each point of interest to a vector on which we can calculate some distances metrics to find the nearest neighbors of one's sentence.



## Model
Creation of a classification batched base learning model.

