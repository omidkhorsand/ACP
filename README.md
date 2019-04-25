# Automated Modelling Pipeline

This project aims to automate feature engineering to increase modelling efficiency and performance. Automated components of the project include Python classes for feature extraction (FeatureExtraction.py) and feature selection (FeatureSelection.py).

## Installation

Install with pip

	python -m pip install featuretools
	python pip install scikit-learn
	python pip install matplotlib
	python pip install pandas
	python pip install numpy

Install with conda

	conda install -c conda-forge featuretools
	conda install scikit-learn
	conda install -c conda-forge matplotlib
	conda install pandas
	conda install numpy
	

## Feature Extraction

The FeatureExtraction class uses Deep Feature Synthesis from Featuretools to automate feature generation using both aggregate and transform functions. In addition, we have incorporated the use of multiple training windows to enable feature extraction relative to a cutoff date and user-defined windows as well as functions to calculate the difference and average rate of change between generated window-based features. Note that entity sets and relationships as it pertains to the dataset need to be set for the class to run. This is covered in the demo, and more information can be found at https://docs.featuretools.com/

## Feature Selection

The input for our Feature Selection pipeline is any Pandas dataframe, including the output from the FeatureExtraction.py. Under FeatureSelection.py we have a few classes to automate feature selection:

* DropDuplicate - drops duplicate columns
* DropMissing - drops columns with missing values above a specified threshold
* DropHighCorr - drops one of every two highly similar columns based on a specified threshold and metric (ex. correlation, cosine, euclidean, and more)
* DropZeroCov - drops columns with a single unique value (zero variance) 
* MISelector - selects the most important features by estimating mutual information for a discrete target variable baed on entropy
* RFSelector - uses a random forest model to select the most important features

The pipeline is based on scikit-learn's pipeline class and can be run similarly. For example:

	RFPipeline = Pipeline([
    		('drop_missing', DropMissing(threshold=0.7)),
    		('drop_duplicate', DropDuplicate()),
    		('drop_zerocov', DropZeroCov()),
    		('drop_correlated', DropHighCorr(threshold=0.9)),
    		('RF_selector', RFSelector(num_feat=0.8))
		])

	processed_data = RFPipeline.fit_transform(sample.df)

## Demos

For detailed examples, we recommend looking at the FeatureExtraction_Demo Jupyter Notebook to learn more about  applying  the FeatureExtraction class. We also recommend looking at the feature_engineering_kkbox notebook to see a full application of the pipeline.

## Testing

Under the test folder in the repository, we have written two test files (test_FeatureSelection.py, test_FeatureExtraction.py) that can be run with pytest. These files are to ensure that any changes made by the user result should result in the same output. 

## Additional Resources

For more information on Featuretools, we recommend visiting: https://www.featuretools.com

For more information on Deep Feature Synthesis, we recommend reading the following paper: https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf

 