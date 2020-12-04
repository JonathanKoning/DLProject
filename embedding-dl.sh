declare -a links=("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip" "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip" "http://nlp.stanford.edu/data/glove.840B.300d.zip" "http://nlp.stanford.edu/data/glove.42B.300d.zip")

mkdir embeddings
cd embeddings

echo
echo "Downloading..."
echo

index=1

for link in "${links[@]}"
do
    echo [${index}/29] [downloading] $link
    wget -c "$link"

    index=$((index+1))

done

echo
echo "Downloads complete!"
echo
echo "Extracting..."
echo

index=1

for file in ./*.zip
do
  echo [${index}/29] [extracting] $file
  unzip "$file"

  index=$((index+1))

done