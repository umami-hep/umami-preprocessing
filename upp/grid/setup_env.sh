# setupATLAS -c el9

asetup "Athena,25.0.36"
lsetup "python 3.11.9"

python3 -m venv env
source env/bin/activate
# Install this framework
pip install -e .
pip install pickleshare
pip install backcall

lsetup panda
lsetup rucio