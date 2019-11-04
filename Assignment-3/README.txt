RUN:  git clone git@github.com:saadwaraich1/CS7641.git
OR
GITHUB REPO: https://github.com/saadwaraich1/CS7641/tree/master

From Assignment-3 folder do all following to get graphs

Setup conda environment using "requirements.yml"

Run following commands:

conda env create --name envname --file=environments.yml
conda activate envname

Below method could be used to get all graphs used in report. All graphs will be saved in "figures/" folder which is already created in repo. All commands need to be run from Assignment-3 folder without any alterations.
To run all files run follwing command.
./run.sh

OR run all files separately using following
python ica.py
python kmem.py
python lda.py
python nn.py
python pca.py
python rp.py


Datsets are provided in repo but if its still needed to be downloaded then it can be found here.

https://www.kaggle.com/uciml/pima-indians-diabetes-database
https://www.kaggle.com/ronitf/heart-disease-uci/version/1#