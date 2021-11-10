# Decoding_NER_Biases

This repository is used for a Sciences Po group project for the course '*Decoding Biases in AI*'.

**Group Members:**

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


### Models: 

For this project, we are testing the hypothesis of biases for different models, all implemented in Spacy. 

These models are:

- en_core_web_sm
  Trained on 
- en_core_web_md
- en_core_web_lg
- en_core_web_trf (model based on roberta)

**Why do we need to test all of these models?**

When we perform a simple test on all of these models, we can clearly see that the results are quite different. 
For instance, on the test text : `I think Barack Obama met the founder of Facebook at the occasion of a release of a new NLP algorithm.`, we obtain the results below for each model: 

#### sm:
![Image](Decoding_NER_Biases/images/sm.JPG)
#### md:

#### lg:

#### trf:


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
2) To access geo datasets - made available inside the package. 


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

### Datasets used for this project: 

#### World Cities Dataset by `simplemaps`

This dataset can be found [here](https://simplemaps.com/data/world-cities). 

The basic (free) one is used.



--------------------
# Draft/examples
-------------------



## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/adrihans/Decoding_NER_Biases/edit/main/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/adrihans/Decoding_NER_Biases/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
