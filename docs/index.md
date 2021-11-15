# Decoding_NER_Biases

This repository is used for a Sciences Po group project for the course '*Decoding Biases in AI*'.

**Group Members:**
> Eleanora BONEL, Ryan HACHEM, Adrien HANS, Sara KEMPPAINEN, Pablo PIGUET




---------------------------

## Methodology and application

In order to look for biases and to test the fairness of NER algorithms, we had to get two main things for each application: a dataset containing what we had to test with detailed information when useful (i.e first names, city names ... with their respective ethnicity, birth year, population...), and a templates with real sentences in which we could test the names entities. Indeed, we had to test the hypothesis on real life sentences. 

With this template, we could then replace the named entities by the ones of the dataframe and get the results by applying the models. 

We describe below the methodology for each application, after explaining what models we are testing the hypothesis on and why we are testing several ones. 


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
![Image](images/sm.JPG)
#### md:
![Image](images/md.JPG)
#### lg:
![Image](images/lg.JPG)
#### trf:
![Image](images/trf.JPG)


### First names 

Firstly, since we were really inspired by the article, but also because we thought they were not going far enough, we tested our hypothesis of existing biases in NER models on first names. 

The template was not hard to find, and we used the winogender one. It consists of ... We had to clean the sentences, here again quite inspring ourself with the article. 

Then, we applied the models on several datasets: 
1. Same first names as in the article 
2. US Baby names
3. NYC opendata

### Geographical biased ? 


Then, we asked ourseleves if we could find the same kind of results on other applications like geographical named entities. We wondered if western geographical named entities were more recognized by non-western ones for instance. 
So we applied the same kind of methodology we used for the first names but on two things : City names and Country names. 

Here, we did not find any available templates for this kind of application. We then had to build them ourselves. 
Instead of writing basic sentences, we thought it would be more accurate to use real-life sentences. In order to do that, we decided to use sentences containing the named entity from wikipedia pages summaries. 

For instance, to get sentences for country names we used Wikipedia API to get wikipedia pages and then summaries. We split the summaries by sentences. Looping on each sentences we looked for sentences containing the country name - of the wikipedia page - and then we replace the country name by `$COUNTRY` to build the template. 

This gave for country names ... sentences. 
Some example of the sentences: 


We used the same kind of method for city names, with ... sentences.


We used two different datasets: 

**For country names**: 

world by Geopandas 

**For city names**:

At the end, we applied the 4 models, using the same validation method we used for the first names. 
Since we are here not really interested in actual results for each city but more global results, we computed the results for country and continent by grouping by the scores. 
This enabled us top plot some maps we are detailing in the `Results` part of this page. 

### Company names 

The same kind of methodology that we described for city and country names have been used for company names. 

The templates have been built up using the same method. 




### Averaging the means. 

---------------------
## Results

### First names


### Geographical named entities
The first geopgraphical named entity test we computed is on country names. 
The results are shown below, with average results on country names by country and by continent. 

![Results on country names by country](images/results/avg_score_country_names.png)

![Results on country names by continent](images/results/avg_score_country_names_by_continent.png)

We can clearly see that there does not exist a real difference between those results.  

Moreover, we are not completly sure about these results from a scientifical standpoint. 

Indeed, the main issue we had with country names is that there is not only one name for each country. For instance, in the `world` dataset from geopandas, the name of USA was 'United States of America', but running a simple test we can clearly see that the results are quite different depending on how the name is implemented. 

![Image of different results depending on the way the country name is computed](images/results/score_us_america_usa.JPG)

We then don't think that the results obtained for country names are not really ... 
This is why we also tested the hypothesis on city names, as described in the methodology part of this page. 

The results on city names are quite convincing that there exists a bias. 
Indeed, if we plot the mean scores by continent, we can clearly see that the best results are obtained for North America, with around 5 points better than Africa, South America and Europe. 

![Image results for city names by continent](images/results/avg_score_country_names_by_continent.png)

Additionally, the same kind of results are obtained for every models. 

### Company names

The same kind of method has been used on company names. 

Possible biases in this field could have great consequences because it would mean that if non-western company names are less recognized by these algorithms than western ones, those companies could end up being less tagged on press article for instance. 

Obtaining results for company names was actually very difficult, for different reasons. Indeed, one should not compare companies which are too different. For instance, just like we saw that californian first names were the most recognized in America, we could think - from who is actually training the models - that tech companies are more recognized than construction ones. The other thing that could bias the search for these biases is the choice of company depending on the size of it. 

### Possible improvements

We have been given restricted time for this project and therefore could not apply everything we wanted but here are some of the main possible improvements :

1. Applying the models to every possible sentences. 

This has been developped inside the notebooks, but given our computational power we could not apply the model to every possible sentences for most of the explored biases. 

Indeed, 

2. Exploring the label of the results. 

The other main improvement could have been to check if the named entity were recognized as what they actually are. 

For instance, a person name could be recognized by the algorithm as a company name, but given our procedure we give a score of one to the algorithm. 

Exploring the label could add different elements to the project. Possible new biases could emerge : are white people first names recognized by the algorithm more recognized as a person names than those of a different ethnicity for instance ? Another metric could also have been implemented. 



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



-------------------
