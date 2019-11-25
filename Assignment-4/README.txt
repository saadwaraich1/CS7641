Exact code link: https://github.com/saadwaraich1/CS7641/tree/master/Assignment-4
RUN:  git clone git@github.com:saadwaraich1/CS7641.git
OR
GITHUB REPO: https://github.com/saadwaraich1/CS7641/tree/master

From Assignment-4 folder do all following to get graphs

Setup conda environment using "requirements.yml"

Run following commands:

conda env create --name envname --file=environments.yml
conda activate envname

Below method could be used to get all graphs used in report. All graphs will be saved in "figures/" folder which is already created in repo. If repo is not cloned then "figures" directory needs to be created. All commands need to be run from Assignment-3 folder without any alterations.


OR run all files separately using following

For Value Iteration:
python vi.py

For Policy Iteration:
python pi.py

For Q-learning
python ql.py
