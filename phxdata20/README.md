# Predicting Wheat Futures with Aerial Imagery
This repository includes materials for an introduction to trading with 3rd party datasets. 

The most recent lecture can be found in the [powerpoint presentation](https://github.com/athaker/econ_136/blob/master/phxdata20/phxdata20_avi_thaker_aerial_imagery-2020-11-20.pdf)

## 1. Getting Started
1. Download or clone the files in the repository to your local machine. (Green button top right). 
2. Decide if you want to run locally (step 2), or run online with Colab. 


## 2. Running Online
The suggested environment is [Google Colab](https://colab.research.google.com), as Prof Evans mentioned this will allow you to have a GPU (likely a Tesla K80) for free, and more quickly run your experiments. 
1. Open: https://colab.research.google.com
2. Upload Notebook: File -> Upload Notebook, and upload the notebook _aerial_imagery/overhead_aerial_imagery.ipynb_. 
3. (OPTIONAL) Start an accelerated runtime: Runtime -> Change Runtime Type -> Hardware accelerator -> GPU
4. Uncomment the below lines in the second code cell to allow you to upload files. Run to that point and upload the two zipped files.
```
from google.colab import files
uploaded = files.upload()
```

## 3. (OPTIONAL) Installing and Running Locally
### Requirements
Pandas, Tensorflow >= 1.13, Numpy

Everything should be pip installable. 
### Run
To run a [Jupyter Notebook](http://jupyter.org/) simply run the command: ```jupyter notebook overhead_aerial_imagery.ipynb```

A [Jupyter Notebook (IPython Kernel)](http://jupyter.org/) is a web based interactive compute environment featuring [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop). This makes iterative learning quick and easy!
