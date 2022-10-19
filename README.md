


## File tree structure

    ├── data                      
    │   ├── raw                   # Folder where we can find the raw files to be taken to create the graph
    │   ├── processed             # Folder for the processed dataset to be used for training our model
    ├── models                    # Folder for stored model
    ├── pytorch-geometric         # Folder for source files
    ├── LICENSE
    └── README.md

# Steps of EDA and embedding

## Embedding for text
We used a transformers to highlight the deep meaning that is contain in the aggregation of name and categories columns.
For that we choose a language agnostic bert based model to transform a concatenation of the name and categories of each point of interest to a vector.


## K-Nearest neighbors graph calculation

### Country based-selection
We choose to have all the nearest neighbors search by country for the machine to be able to compute in an acceptable time for metrics to be calculated in a smaller set. We calculated that around only 0.002% of true links are lost and could not be found afterward while performing edges classification.

### Nearest neighbors on embeddings
The embedding associated with each "text" column ("name" + "categories" columns concatenated) is used to compute a 'cosine' distance (1-{cosine similarity score}) which will allow using a the k-nearest neighbors algorithm to create a graph.  
We also choose to get a threshold on distance above which we will not select the neighbor given by the k-nearest neighbors algorithm. For that we focused on the known duplicates POI in a random selected part of the half of the training set and took out some statistics :

| Measures | Values |
| -------- | ----------- |
| count | 256461.000000 |
| mean | 0.175741 |
| std | 0.129458 |
| min | 0.000000 |
| 25% | 0.077977 |
| 50% | 0.157808 |
| 75% | 0.256179 |
| max | 0.881651 |


### Nearest neighbors on distance
The latitude and longitude value of each POI is used to compute a haversine distance which will allow using a the k-nearest neighbors algorithm to create a graph.  
We also choose to get a threshold on distance above which we will not select the neighbor given by the k-nearest neighbors algorithm. For that we focused on the known duplicates POI in a random selected part of the half of the training set and took out some statistics :

| Measures | Values |
| -------- | ----------- |
| count | 256461.000000 |
| mean | 85.686376 |
| std | 351.627479 |
| min | 0.000000 |
| 25% | 0.074512 |
| 50% | 0.592190 |
| 75% | 7.067702 |
| max | 8984.622473 |


## Model
Creation of a classification batched base learning model.

