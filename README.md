# Decoding_NER_Biases

This repository is used for a Sciences Po group project for the course '*Decoding Biases in AI*'.

The associated github page can be accessed [here](https://adrihans.github.io/Decoding_NER_Biases/).

## Group Members:

>Eleonora BONEL, Ryan HACHEM, Adrien HANS, Sara KEMPPAINEN, Pablo PIGUET

## Goals of the project: 

The project is described with more precision on [the GitHub page](https://adrihans.github.io/Decoding_NER_Biases/) and in the notebooks, but it consists in exploring the possible biases in Named Entity Recognition (NER) models.  

To do so, we explore three main named entities, with specific possible biases: 

1. First names
  - Ethnical popularity ?
  - Birthyear popularity ? 
  - Geographical ? 
  - Gender ? 
3. Geographical named entities
  - City
  - Country
5. Company names
  - Western vs non-western
  - Size ?

The main inspiration for this project comes from [this article](https://arxiv.org/pdf/2008.03415.pdf), but we thought researchers were not going far enough in their exploration - they were only testing biases depending on gender and ethnical groups. Moreover, the way they defined the first names was not really satisfying. Indeed, we were not satisfied with the fact they defined specific first names for each ethnicity while we know that there is no such thing as first names exclusively associated with one ethnicity.
Therefore, as a first step for the whole project and particurally for the *first names* part, we followed the exact same procedure as them, but we then tried to go further using additional datasets, mentionned below or in more detailed on [the GitHub page](https://adrihans.github.io/Decoding_NER_Biases/). 

## Content of this repository

### Notebook

This repo consists in four main notebooks : 

1. [basic_NER_models_comparison.ipynb](https://github.com/adrihans/Decoding_NER_Biases/blob/main/basic_NER_models_comparison.ipynb)

Exploring the models integrated in Spacy with tests on a simple sentence. 

1. [complete_exploration_first_names.ipynb](https://github.com/adrihans/Decoding_NER_Biases/blob/main/complete_exploration_first_names.ipynb)

Exploring the possible biases with NER algorithms depending on the models and on several points like ethnicity, gender, age...

3. [exploration_geographical.ipynb](https://github.com/adrihans/Decoding_NER_Biases/blob/main/exploration_geographical.ipynb)

Exploring the possible geographical biases, testing geographical named entities like city or country names. 

4. [exploration_companies.ipynb](https://github.com/adrihans/Decoding_NER_Biases/blob/main/exploration_companies.ipynb)

Exploring the possible biases with company names. 

### Additional content

The folder contains the datasets used, when they were not too heavy to post them on github. 

## Main results




## Technical information

### Installations

#### Spacy (v3.0 or superior)

Spacy v3.1 is used to complete the task. 
Following the [Spacy installation guide](https://spacy.io/usage), this is the lines you need to enter to install Spacy: 

```
pip install -U pip setuptools wheel
pip install -U spacy
```

Note that version 3.0 or superior is needed to use the transformer model.

To install the models inside spacy: 

```
python -m spacy download en_core_web_sm
```

You just have to replace the model name by the one you want to install. A list of the models avaiblable in Sapcy is available [here](https://spacy.io/usage/models).

#### Wikipedia package

To install the wikipedia package, you cas use pip, following this [installation guide](https://pypi.org/project/wikipedia/).

```
pip install wikipedia 
```

This package is helping us to build templates for the sentences, in order to use real sentences to test the model. 


#### Geopandas 

Geopandas is useful when dealing with geographic datasets. 

It is used inside this project :
1) To plot some maps 
2) To access geo datasets made available inside the package. 


You can also use pip to install it : 
```
!pip install --upgrade geopandas
```
If using Google collab, you may want to add these lines - and these packages - to make it work : 
```
!pip install --upgrade pyshp
!pip install --upgrade shapely
!pip install --upgrade descartes
```

### Complete list of the datasets used for this project:

#### Winogender 

The dataset we used the most is the winogender schemas. It consists
It can be accessed on [this Github Page](). 

Specifically, the one we used is [this one](). 


#### US Baby names

We used two versions of the US Baby names datasets: ([nation]) and ([state]). 
It made us able to explore biases depending on the state and the year people were born in. 

#### NYC 
