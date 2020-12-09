Balanced review counts

Appliances                         45
Gift Cards                         55
All Beauty                        318
Kindle Store                      320
Fashion                           465
Magazine Subscriptions            510
Software                         3595
Luxury Beauty                    5475
Digital Music                    9060
Inductrial and Scientific        9680
Prime Pantry                    16270
Musical Instruments             36385
Arts Crafts and Sewing          63920
Office Products                120380
Video Games                    120675
Patio Lawn and Garden          161960
Grocery and Gourmet Food       210660
CDs and Vinyl                  217475
Automotive                     260450
Cell Phones and Accessories    285875
Toys and Games                 291310
Tools and Home Improvement     364550
Sports and Outdoors            508185
Pet Supplies                   538705
Movies and TV                  862195
Home and Kitchen              1447320
Electronics                   1533380
Clothing Shoes and Jewelry    2723540



# Embeddings

## all-87k-300d.vec

 paths = [
    "data/csv/all-beauty-test.csv",
    "data/csv/all-beauty-train.csv",
    "data/csv/grocery-and-gourmet-food-test.csv",
    "data/csv/grocery-and-gourmet-food-train.csv",
    "data/csv/home-and-kitchen-test.csv",
    "data/csv/home-and-kitchen-train.csv",
    "data/csv/office-products-train.csv",
    "data/csv/office-products-test.csv",
    "data/csv/pet-supplies-train.csv",
    "data/csv/pet-supplies-test.csv",
    "data/csv/prime-pantry-train.csv",
]

word2vec = Word2Vec(
    sentences,
    sg= 1,
    size= 300,         # Dimension of the word embedding vectors
    window= 5,    # Radius of skip-gram / cbow window from current word
    min_count= 5,
    iter= 5
)

## all-119k-300d.vec

paths = [
    "data/csv/all-beauty-test.csv",
    "data/csv/all-beauty-train.csv",
    "data/csv/grocery-and-gourmet-food-test.csv",
    "data/csv/grocery-and-gourmet-food-train.csv",
    "data/csv/home-and-kitchen-test.csv",
    "data/csv/home-and-kitchen-train.csv",
    "data/csv/office-products-train.csv",
    "data/csv/office-products-test.csv",
    "data/csv/pet-supplies-train.csv",
    "data/csv/pet-supplies-test.csv",
    "data/csv/prime-pantry-train.csv",
]

word2vec = Word2Vec(
    sentences,
    sg= 1,
    size= 300,         # Dimension of the word embedding vectors
    window= 5,    # Radius of skip-gram / cbow window from current word
    min_count= 3,
    iter= 5
)

## food-36k-300d

paths = [
    "data/csv/grocery-and-gourmet-food-test.csv",
    "data/csv/grocery-and-gourmet-food-train.csv",
    "data/csv/prime-pantry-train.csv",
]

word2vec = Word2Vec(
    sentences,
    sg= 1,
    size= 300,         # Dimension of the word embedding vectors
    window= 5,    # Radius of skip-gram / cbow window from current word
    min_count= 3,
    iter= 5
)