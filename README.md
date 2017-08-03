# kaggle-planet-forest
My solution to the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). I have written a [post](https://medium.com/towards-data-science/my-first-kaggle-competition-9d56d4773607) describing this solution. So far, I have not cleaned up my code enough to release the full 5-fold cross validation solution. This code is for a single model, using the first 8K labelled images for cross validation, and the rest of the 32K labelled images for training. Use `git log` to see the history of this code.

## Pre-requisites and downloads
- Install [PyTorch](http://pytorch.org/) with Python 3.
- We shall reference the root directory where `my_solution.py` as `./`
- Download this [pre-trained model](https://download.pytorch.org/models/resnet34-333f7ec4.pth) and rename it as `./resnet34_pretrained.pth`.
- Make a directory `./data/` and download the data from the [competition's website](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data) into this directory. The data directory should have something like the following tree structure:
```
./data/sample_submission_v2.csv
./data/test-jpg
./data/test-jpg-additional
./data/test-tif
./data/test-tif-v2
./data/train-jpg
./data/train-jpg-sample
./data/train-tif
./data/train-tif-sample
./data/train-tif-v2
./data/train.csv
```

## Running the solution
To run the solution, use the following command in terminal:
```
python3 my_solution.py
```
Most of the important parameters are after the line `if __name__ == "__main__":`. In particular, give the experiment an index using the `experiment` variable. The results are stored in the relevant `./results/experiment<index>/`.

To view the learning curves while training, use the following command in terminal:
```
python3 plot_learning_curves_single.py
```
