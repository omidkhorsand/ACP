import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances 
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


class DropDuplicate(BaseEstimator, TransformerMixin):
    """
    Drops duplicate columns
    
    """
        
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """
        Argument: 
        X: Pandas Dataframe
        
        fit method identifies unique columns and stores them in keepList
        
        """
        
        # store original dtypes to a dictionary to preserve dtypes
        dtype_dict = X.dtypes.to_dict()
        
        # drop duplicate columns
        result = X.T.drop_duplicates().T
        
        # remove dropped column names from the dtype dictionary 
        for col in X.columns:
            if col not in result.columns:
                del dtype_dict[col]
        
        # set attributes 
        self.dTypes_ = dtype_dict
        self.keepList_ = result.columns
        self.numDuplicates_ = len(X.columns) - len(result.columns)
        
        return self
    
    def transform(self, X):
        """
        Transforms X by dropping duplicate columns 
        
        """
        print("--->", self.numDuplicates_, "columns dropped due to duplicate values:\n")
        
        # subset to keepList
        X = X.loc[:, self.keepList_]
        
        # convert dtypes 
        return X.astype(self.dTypes_)



class DropMissing(BaseEstimator, TransformerMixin):
    """
    Drops columns with missing values ratio above a threshold
    
    """
    
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """
        Argument: 
        X: Pandas dataframe
        
        """
        
        # find missing percentage for each column
        missing_pct = X.isnull().sum() / len(X)
        
        # identify columns with missing ratio above the threshold
        to_drop = list((missing_pct[missing_pct >= self.threshold]).index)
        
        # define attribute for list of columns to drop
        self.dropList_ = to_drop
        
        # define attribute for the number of columns to be dropped
        self.numMissingCol_ = len(self.dropList_)
        
        return self
    
    def transform(self, X):
        """
        Transforms X by dropping high missing columns
        
        """

        print("--->", self.numMissingCol_, "columns dropped due to missing values:", self.dropList_)

        # return a dataframe with dropped columns missing over threshold 
        return X.drop(columns=self.dropList_)


class DropHighCorr(BaseEstimator, TransformerMixin):
    """
    Drops one of every two similar columns
    
    Acceptable metrics are: 
    
    ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
    ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, 
    ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, 
    ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
    ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
    
    """
    
    def __init__(self, threshold=0.9, metric = None):
        self.threshold = threshold
        self.metric = metric
    
    def fit(self, X, y=None):
        """
        Argument: 
        X: Pandas dataframe
        
        """        
        
        if self.metric is None:
            # Calculate correlation matrix
            sim_matrix = X.corr().abs()
        else:
            # calculate a similarity matrix
            sim_matrix = pairwise_distances(X.t, metric)

        # Subset to upper triangle of sim_matrix
        upper = sim_matrix.where(
            np.triu(np.ones(sim_matrix.shape), k=1).astype(np.bool))

        # Identify columns with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] >= self.threshold)]
        
        # define attribute for list of columns to drop
        self.dropList_ = to_drop
       
        # define attribute for the number of columns to be dropped
        self.numDropCols_ = len(self.dropList_)
        
        return self
    
    def transform(self, X):
        """
        Transforms X by dropping high missing columns
        
        """
        
        print("--->", self.numDropCols_, "columns dropped due to collinearity:\n", self.dropList_)

        # return a dataframe with dropped columns
        return X.drop(columns=self.dropList_)



class DropZeroCov(BaseEstimator, TransformerMixin):
    """
    Drops columns with a single unique value (zero variance)
    
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """
        Argument: 
        X: Pandas dataframe
        
        """                
        one_unique = X.apply(lambda x: x.nunique() == 1, axis=0)
        to_drop = list(one_unique[one_unique == True].index)
        
        self.dropList_ = to_drop
        self.numZeroCov_ = len(self.dropList_)
        
        return self
    
    def transform(self, X):
        """
        Transforms X by dropping columns with zero variance
        
        """
        print("--->", self.numZeroCov_, 'columns dropped due to zero variance:\n', self.dropList_)

        # return a dataframe without columns with a single unique value 
        return X.drop(columns=self.dropList_)


class MISelector(BaseEstimator, TransformerMixin):
    """
    Selects most important features by estimating mutual information for a discrete target variable based on entropy
    num_feat can be given as an integer or as a float
    If num_feat is given as an integer e.g. 10, it will be used to select top 10 features 
    If given as a float e.g. 0.8, it will be used to select enough features to cover 80% of total mutual importance
    
    """
    
    def __init__(self, num_feat=100):
        self.num_feat = num_feat
        """
        Arguments:
        num_feat: is an integer if user specifies the number of features or a float between 0 and 1

        """
    def fit(self, X, y=None):
        """
        Using Scikit-learn's mutual_info_classif to estimate mutual information between each feature and the target.
        
        Arguments:
        X: 2D Pandas dataframe
        y: 1D Pandas dataframe
        
        """
        
        # Impute mean for missing values
        X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
        
        # calculate mutual information
        mi = mutual_info_classif(X, y)
        # normalize 
        self.mi_ = mi/sum(mi)
        
        # Identify top features
        dic = dict(zip(X.columns, self.mi_))
        # ordered list of features 
        self.features_ordered_ = [k for k in sorted(dic, key=dic.get, reverse=True)]
        # ordered list based on entropy
        self.mi_ordered_ = [dic[k] for k in self.features_ordered_]

        if isinstance(self.num_feat, int):
            # if user selected a specific number of fearures 
            self.top_feature_list_ = self.features_ordered_[:self.num_feat]
        else:
            # if the user specified a threshold of overall imporatance 
            cumsum = np.cumsum(self.mi_ordered_)
            self.num_feat_selected_ = np.argmax(cumsum>=self.num_feat) + 1
            self.top_feature_list_ = self.features_ordered_[:self.num_feat_selected_]
            
        return self
    
    def transform(self, X):
        """
        Returns a reduced Dataframe
        
        """
    
        return X[self.top_feature_list_]
    
    def plot_importance(self, n=10):
        
        """
        plots important features
        
        Arguments: 
        n: numner of features to plot
        """
        
        ft = self.features_ordered_.copy()
        mi = self.mi_ordered_.copy()
        df = pd.DataFrame({'importance_normalized':mi, 'feature': ft})
        df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                                x = 'feature', color = 'blue', 
                                edgecolor = 'k', figsize = (12, 8),
                                legend = False)

        plt.xlabel('Normalized Information Gain', size = 18); plt.ylabel(''); 
        plt.title('Top %d Most Important Features'%n, size = 18)
        plt.gca().invert_yaxis()


class RFSelector(BaseEstimator, TransformerMixin):
    """
    Selects most important features by training a random forest model 
    num_feat can be given as an integer or as a float
    If num_feat is given as an integer e.g. 10, it will be used to select top 10 features 
    If given as a float e.g. 0.8, it will be used to select enough features to cover 80% of total feature importance
    
    """
    
    def __init__(self, num_feat=100):
        self.num_feat = num_feat
        """
        Arguments:
        num_feat: is an integer if user specifies the number of features or a float between 0 and 1 if user 

        """
    def fit(self, X, y=None):
        """
        Fit a Random Forest model to data to find most important features
        
        Arguments:
        X: 2D Pandas dataframe
        y: 1D Pandas dataframe
        
        """
        
        # Impute mean for missing values
        X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
        
        # Random Forest model
        rf = RandomForestClassifier()
        rf.fit(X, y)
        
        # Identify top features
        dic = dict(zip(X.columns, rf.feature_importances_))
        self.features_ordered_ = [k for k in sorted(dic, key=dic.get, reverse=True)]
        self.importance_ordered_ = [dic[k] for k in self.features_ordered_]

        if isinstance(self.num_feat, int):
            # if user selected a specific number of fearures 
            self.top_feature_list_ = self.features_ordered_[:self.num_feat]
        else:
            # if the user specified a threshold of overall imporatance 
            cumsum = np.cumsum(self.importance_ordered_)
            self.num_feat_selected_ = np.argmax(cumsum>=self.num_feat) + 1
            self.top_feature_list_ = self.features_ordered_[:self.num_feat_selected_]
            
        return self
    
    def transform(self, X):
        """
        Returns a reduced Dataframe
        
        """
    
        return X[self.top_feature_list_]

    def plot_importance(self, n=10):
        
        """
        plots important features
        
        Arguments: 
        n: numner of features to plot
        """
        
        ft = self.features_ordered_.copy()
        im = self.importance_ordered_.copy()
        df = pd.DataFrame({'importance_normalized':im, 'feature': ft})
        df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                                x = 'feature', color = 'blue', 
                                edgecolor = 'k', figsize = (12, 8),
                                legend = False)

        plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
        plt.title('Top %d Most Important Features'%n, size = 18)
        plt.gca().invert_yaxis()


RFPipeline = Pipeline([
    ('drop_missing', DropMissing(threshold=0.7)),
    ('drop_duplicate', DropDuplicate()),
    ('drop_zerocov', DropZeroCov()),
    ('drop_correlated', DropHighCorr(threshold=0.9)),
    ('RF_selector', RFSelector(num_feat=0.8))
])