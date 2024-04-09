URL="https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download"

mkdir -p data

wget -O data/ACDC.zip $URL
unzip -q data/ACDC.zip -d data/
rm data/ACDC.zip
