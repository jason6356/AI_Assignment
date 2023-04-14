@echo off
Rem Checking the Python Version
python --version
echo "Setting Up The Libraries..."
pip install -r requirements.txt

echo "Setting Up The Packt from NLTK..."
python ./nltkInit.py

echo "Checking for Updates for SKLearn"
pip install -U scikit-learn

echo "Project Successfully Initialized"