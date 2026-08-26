set -e # interrompe lo script al primo errore

echo "Download del dataset da Kaggle..."
kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 -p data/raw --unzip
echo "Download completato! I file sono stati salvati in data/raw."