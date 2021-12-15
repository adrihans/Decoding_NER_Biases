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

----------------------

### Additional content

The folder contains the datasets used, when they were not too heavy to post them on github.

---------------------------
## Technical information

----------------------------------------

### Github and notebooks

The complete code for this project is accessible on the associated Github Repo here. Some data could not have been posted in the repo because they were too large, but the complete list of datasets used for this project is written below. 

The Repo contains four main notebooks:


------------------------------------

### Installations

|Package name|Description and ways to install it|Doc link|
|------------|----------------------------------|--------|
|Spacy (v3.0 or superior)|Spacy v3.1 is used to complete the task. Following the [Spacy installation guide](https://spacy.io/usage), this is the lines you need to enter to install Spacy: <br/>```pip install -U pip setuptools wheel``` <br/> ```pip install -U spacy ```<br/>Note that version 3.0 or superior is needed to use the transformer model.To install the models inside spacy: <br/> ```python -m spacy download en_core_web_sm```<br/>You just have to replace the model name by the one you want to install. <br/> A list of the models available in Sapcy is available [here](https://spacy.io/usage/models).|[Spacy Documentation](https://spacy.io/)|
|Wikipedia|To install the wikipedia package, you cas use pip, following this [installation guide](https://pypi.org/project/wikipedia/).<br/>```pip install wikipedia```<br/>This package is helping us to build templates for the sentences, in order to use real sentences to test the model.|[Wikipedia API documentation for Python](https://pypi.org/project/wikipedia/)|
|Geopandas|Geopandas is useful when dealing with geographic datasets.<br/>It is used inside this project :<br/>1) To plot some maps <br/>2) To access geo datasets - made available inside the package. <br/>You can also use pip to install it : <br/>```!pip install --upgrade geopandas```<br/>If using Google collab, you may want to add these lines - and these packages - to make it work : <br/>```!pip install --upgrade pyshp```<br/>```!pip install --upgrade shapely```<br/>```!pip install --upgrade descartes```<br/>|[Geopandas Documentation](https://geopandas.org/)|

#### Other packages

We also used other packages available in Python, but quite common ones. 

For instance, we used `tqdm` to be able to check the process and its final time. 

-------------------------

### :file_cabinet: Datasets used for this project: 

|Field|Name|Description|Source|
|-----|----|-----------|------|
|First names|list of first names given in the article|The dataset consisted in 15 or 16 first names for each ethnical and gender category. <br /> This dataset was completely described in the above `methodology` part of this page. |[Article](https://arxiv.org/pdf/2008.03415.pdf)|
|First names|`Winogender Schemas`|This dataset provided us with a sentence template. <br /> Each sentence contained three possible entities: `OCCUPATION`, `PARTICIPANT` and a `PRONOUN`. <br /> After cleaning the dataset to avoid mistakes, it enabled us to build 89 correct sentences to test the models on. <br /> The specific dataset we used was [this one](https://github.com/rudinger/winogender-schemas/blob/master/data/templates.tsv). |[Winogender-Schemas on Github]()|
|First names|`US baby names` - nation|We used the dataset at the national level to compute a score for each year.| Transformed version on [Kaggle](https://www.kaggle.com/kaggle/us-baby-names). Raw one coming from Data.gov|
|First names|`US baby names` - states|The dataset at the state level was used to compute the scores of the models depending on the state people with each first names were born in. <br /> This allowed us to compute US maps as shown in the results part of this page.  |Transformed version on [Kaggle](https://www.kaggle.com/kaggle/us-baby-names). Raw one coming from Data.gov|
|First names|`Popular Baby Names`|This dataset consists in information about babies born in New York City between 2012 and 2017. <br /> It contains information about Child's First Names, Mother's Ethnicity, Year of Birth, Gender and count if there are individuals with the exact same characteristics. |[NYC Open Data](https://data.cityofnewyork.us/Health/Popular-Baby-Names/25th-nujf)|
|Additional dataset for first names|`States 21Basic`|This dataset was used to access to the geometry of each US state to plot the results on a map.|[Socrata](https://alicia.data.socrata.com/Government/States-21basic/jhnu-yfrj)|
|Additional dataset for first names|`Median Income for each state`|This dataset from the US Census Bureau contains the median income in dollar for each US state.|[United States Census Bureau](https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-income-households.html)|
|Additional dataset for first names|`GDP per capita for each US state`|The dataset consists in the GDP per capita for each US state from 1997 to 2015, in dollars.|[BEA on data.world](https://data.world/gmoney/gdp-per-capita-by-state)|
|Geographical named entities|`naturalearth_lowres`|The dataset consists in a list of the countries with their associated countours, continent and estimated population. <br />This dataset can be accessed symply by computing this code: `world = geopandas.read_file( gpd.datasets.get_path( 'naturalearth_lowres' ))` when Geopandas is installed in the distribution|[Geopandas](https://geopandas.org/en/stable/)|
|Geographical named entities|`World Cities Database`|The dataset used for this project was the basic - and thus free - one. <br /> It consists in prominent cities (large, capitals etc.), with around 40 thousands entities. <br /> There are also the GPS coordinates of the city, its associated country and population.|[simplemaps](https://simplemaps.com/data/world-cities)|
|Companies|`7+ Million Company Dataset`|This dataset provided us with a very large list of companies, with some additional information, such as their specific industry, their size, their country or their estimated number of employees. <br /> Given the size of this dataset (around 1.09GB), it has not been posted on GitHub |[People Data Labs](https://www.kaggle.com/peopledatalabssf/free-7-million-company-dataset)|
