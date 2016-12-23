
#Reducing-Manufacturing-Failures
### Reducing Manufacturing Failures -  A Kaggle Challenge
[Competition Link](https://www.kaggle.com/c/bosch-production-line-performance)

[![Video Description](https://img.youtube.com/vi/GmUUIvj571Q/0.jpg)](https://www.youtube.com/watch?v=GmUUIvj571Q)
---

# Scripts for data exploration, feauture engineering, classifier training and testing


---
* idMovements.ipynb: Visualization of the first 9999 jobs as they progress through the production lines.
	* idMovements.ipynb: Python version of idMovements.ipynb

* visualize1.ipynb: Visualization of categorical features and frequency of features with respect to stations, and lines
	* visualize1.py: Python version of visualize1.ipynb

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



