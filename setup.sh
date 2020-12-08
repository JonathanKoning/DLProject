# Download amazon json data
declare -a links=(
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz"
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz"
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Home_and_Kitchen_5.json.gz"
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Office_Products_5.json.gz"
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Pet_Supplies_5.json.gz"
  "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry_5.json.gz"
)

mkdir data
cd data
mkdir json
cd json

echo
echo "Downloading..."
echo

index=1

for link in "${links[@]}"
do
    echo [${index}/28] [downloading] $link
    wget -c "$link"

    index=$((index+1))

done

echo
echo "Downloads complete!"
echo
echo "Extracting..."
echo

index=1

for file in ./*.json.gz
do
  echo [${index}/28] [extracting] $file
  gzip -d "$file"

  index=$((index+1))

done

# Make a folder for the generated .csv files
mkdir ./data/csv

# Generate all the datasets
python datasets.py