# Deep Learning for Timeseries - Lecture 2018
This repository includes materials for a lecture for Gary Evans 136 Class. 

## 1. Getting Started - Lecture 2018
1. Download or clone the files in the the repository to your local machine. (Green button top right). 
2. Decide if you want to run locally (step 2), or run in an online Jupyter notebook (step 3 - recommended). 

## 2. Installing and Running Locally
### Requirements
Pandas, Tensorflow >= 1.4, Numpy

Everything should be pip installable. However a conda yaml env is provided for convenience. 
### Run
To run a [Jupyter Notebook](http://jupyter.org/) simply run the command: ```jupyter notebook pres_election_2016.ipynb```

A [Jupyter Notebook (IPython Kernel)](http://jupyter.org/) is a web based interactive compute environment featuring [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop). This makes iterative learning quick and easy!

## 3. Running Online
The suggested environment is [Google Colab](https://colab.research.google.com), as Prof Evans mentioned this will allow you to have a GPU (likely a Tesla K80) for free, and more quickly run your experiments. 
1. Open: https://colab.research.google.com
2. Upload Notebook: File -> Upload Notebook, and upload the notebook _pres_election_2016.ipynb_. 
3. Start an accelerated runtime: Runtime -> Change Runtime Type -> Hardware accelerator -> GPU
4. Uncomment the below lines in the second code cell to allow you to upload files. Run to that point and upload the two csv files.
```
#from google.colab import files
#uploaded = files.upload()
```
