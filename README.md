# Decoding_NER_Biases

This repository is used for a Sciences Po group project for the course '*Decoding Biases in AI*'.

## Group Members:

- Eleanora BONEL
- Ryan HACHEM
- Adrien HANS
- Sara KEMPPAINEN
- Pablo PIGUET

## Goals of the project: 

We want to explore the existing biases in NER (Named-Entity Recognition) models. 

Those biases can be based on :
- Gender
- Ethnicity
- Area for city names
- ...

The main goal is to analyze those biases through several datasets and tests. 

The main inspiration for this project comes from [this article](https://arxiv.org/pdf/2008.03415.pdf).


### Models to be tested: 

- en_core_web_sm
- en_core_web_md
- en_core_web_lg
- en_core_web_trf (model based on roberta)


### Steps to be followed:

- Constructing the dataframes 
  - Sentence templates
  - Content to be tested on the models
- Testing the newly built sentences 
- Analyzing the results 
- re-training the algorithms to see if the biases can be reduced

## Areas of study for the biases in NER algorithms: 
### Gender and Ethnicity biases when applying a NER model on first names:

### Location biases when applying a NER model on city names -given the country or the continent:



## Technical information

### Installations

Spacy v3.1 is used to complete the task. 
Following the [Spacy installation guide](https://spacy.io/usage), this is the lines you need to enter to install Spacy: 

'''
pip install -U pip setuptools wheel
pip install -U spacy
'''

Note that version > 3.0 is needed to use the transformer model.

To install the models inside spacy: 

'''
python -m spacy download en_core_web_sm
'''

You just have to replace the model name by the one you want to install. A list of the models avaiblable in Sapcy is available [here](https://spacy.io/usage/models).



### Datasets
