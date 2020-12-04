# DL Project

## Setup

### Data

Run [amazon-dl.sh](/amazon-dl.sh) to download the [amazon dataset](http://jmcauley.ucsd.edu/data/amazon/index_2014.html) (excludes Books reviews).

Acquire the reddit dataset yourself.

`data/` should look like:

```
data
├── amazon
│   ├── AMAZON_FASHION_5.json
│   ├── All_Beauty_5.json
│   ├── Appliances_5.json
│   ├── Arts_Crafts_and_Sewing_5.json
│   ├── Automotive_5.json
│   ├── CDs_and_Vinyl_5.json
│   ├── Cell_Phones_and_Accessories_5.json
│   ├── Clothing_Shoes_and_Jewelry_5.json
│   ├── Digital_Music_5.json
│   ├── Electronics_5.json
│   ├── Gift_Cards_5.json
│   ├── Grocery_and_Gourmet_Food_5.json
│   ├── Home_and_Kitchen_5.json
│   ├── Industrial_and_Scientific_5.json
│   ├── Kindle_Store_5.json
│   ├── Luxury_Beauty_5.json
│   ├── Magazine_Subscriptions_5.json
│   ├── Movies_and_TV_5.json
│   ├── Musical_Instruments_5.json
│   ├── Office_Products_5.json
│   ├── Patio_Lawn_and_Garden_5.json
│   ├── Pet_Supplies_5.json
│   ├── Prime_Pantry_5.json
│   ├── Software_5.json
│   ├── Sports_and_Outdoors_5.json
│   ├── Tools_and_Home_Improvement_5.json
│   ├── Toys_and_Games_5.json
│   └── Video_Games_5.json
└── reddit
    └── comments.json
```


### Embeddings

Run [embedding-dl.sh](/embedding-dl.sh) to download the pretrained GLoVe and Fasttext embeddings.

You will need `python` to run a version of python with gensim installed.

If this is not the case, comment out the python line in the shell script, and just run `convertGloveFiles.py` yourself after the shell script has finished.

You should now have this:

```
embeddings
├── crawl-300d-2M.vec
├── glove.42B.300d.vec
├── glove.840B.300d.vec
└── wiki-news-300d-1M.vec
```

## Organization

### Research and Links to Datasets

[Google Doc](https://docs.google.com/document/d/1DIZNSjwDOl5LSKwPArPg-4PXf9iEjDMwh9uDD-k2jO8/edit?usp=sharing)


### Colab

Will use for training. Will update when we have more info...

## Tasks

#### Word Embedding
- Create word embedding data loader.
- Create skipWord model.
- Create word2vec training loop.
- Figure out how to save and load vocabs and word embeddings.
- 

#### Amazon LSTM

- Give 0-5 stars as context and predict next word
- Predict star rating?
- Some sort of analysis between departments
-

#### Reddit Comments

- Give subreddit as context and predict next word
- Predict Karma? Include prior comment?
- 
