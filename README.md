# Reducing-Manufacturing-Failures
## Reducing Manufacturing Failures -  A Kaggle Challenge
[Link to Competition:](https://www.kaggle.com/c/bosch-production-line-performance)

[![Video Description](https://img.youtube.com/vi/GmUUIvj571Q/0.jpg)](https://www.youtube.com/watch?v=GmUUIvj571Q)

---
### Directory Structure:

* data/ : Holds the training and testing dataset, please download the datasets unzipp them to store in the data folder as csv files


* plots/ : Holds the different visualization plots

* scripts/ : Holds runnable .ipynb, .r and .py files, which are explained below.

---
* idMovements.ipynb: Visualization of the first 9999 jobs as they progress through the production lines.
	* idMovements.ipynb: Python version of idMovements.ipynb

* visualizer.ipynb: Visualization of categorical features and frequency of features with respect to stations, and lines
	* visualizer.py: Python version of visualizer.ipynb

* graph-viz.ipynb: Visualizing the first 1000 Defective Ids and first 1000 Non-Defective Ids using [IBM System G](http://systemg.research.ibm.com/) and their movements across the production lines to develop useful insights about the data.
	* gshell.txt: Text file containing the commands used in graph-viz.ipynb
	* graph-viz.py: Python version of graph-viz.ipynb


* nextProb.ipynb: Finding out the probabilities of seeing a defect after a defect
	* nextProb.py: Python version of nextProb.ipynb

* FeatureSelection.ipynb:  Open using the jupyter notebook. Holds the code for figuring out the top most important features, on which the classification algorithms depends upon.
	* FeatureSelection.py: Python version of FeatureSelection.ipynb

* spark.ipynb:  Open using Databricks Platform/Py-spark. It holds the code for developing the RandomForest Classifier on the chosen subset of important features. At the "REPLACE_YOUR_FILE" location, please provide the path to you test_numeric.csv file on databricks/local machine
	* spark.py: Python version of spark.ipynb

* train.ipynb: Open using Jupyter Notebook. It holds the code and visualizations for developing the different classification algorithms (LibSVM, RBF SVM, Naive Bayes, Random Forest, Gradient Boosting) on the chosen subset of important features
	* train.py: Python version of train.ipynb


----

### Instructions for Installation

* Download Dataset from [Link](https://www.kaggle.com/c/bosch-production-line-performance/data) and unzip the files in the data/ folder
* Run any of the scripts from the scripts/ folder according to the task desired


####  Dependencies 

* Python: 2.7.6
* Pandas: 0.19.1
* Python Sklearn: 0.18.1
* Numpy: 1.8.2
* R: 3.3.2
* py-spark: 2.0.1 or Account over [Databricks](https://databricks.com)
* System-G: (systemg-tools-1.4.0) [For Visualizations] 
* jupyter notebook : 4.2.1 with support for R and Python kernels

The code has been tested on Ubuntu 14.04 LTS system. It should work well on other distributions but has not yet been tested.

----

In case of any issue with installation or otherwise, please contact: [Aayush Mudgal](https://mudgal.ml)


