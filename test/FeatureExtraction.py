import pandas as pd 
import numpy as np
import featuretools as ft
from sklearn.base import BaseEstimator, TransformerMixin

class DropDuplicate(BaseEstimator, TransformerMixin):
    """
    Drops duplicate columns. Used in FeatureExtraction class.
    
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
              
        
        return X.astype(self.dTypes_)
    
class FeatureExtraction():
    '''
    Implements user-defined training windows relative to cutoff times on 
    designated target entities. Requires pre-defined featuretools entity sets 
    and relationships.
    '''
    def __init__(self, EntitySet):
        self.es = EntitySet
        self.agg_primitives = []
        self.trans_primitives = []
        self.where_primitives = []
        self.df = None
        self.feature_defs = []
    
    def add_agg_primitives(self, agg):
        '''Appends items from agg to the aggregate primitives to be used
           in DFS. Aggregate primitives must be available in Featuretools
           library.
           
           agg: list of string values
        '''
        aggs_list = ft.list_primitives().loc[ft.list_primitives()['type'] == 
                                  'aggregation']
        for i in agg:
            if i in aggs_list['name'].values:
                self.agg_primitives.append(i)
            else:
                print(i, "is not in the available aggregate primitives.")
        print("The aggregate primitives have been added: ", 
              *self.agg_primitives)
    
    def add_trans_primitives(self, trans):
        '''Appends items from trans to the transformative primitives to be used
           in DFS. Transform primitives must be available in Featuretools
           library.
           
           trans: list of string values
        '''
        tran_list = ft.list_primitives().loc[ft.list_primitives()['type'] == 
                                  'transform']
        for i in trans:
            if i in tran_list['name'].values:
                self.trans_primitives.append(i)
            else:
                print(i, "is not in the available transform primitives.")
        print("The transformative primitives have been added: ", 
              *self.trans_primitives)
    
    def add_where_primitives(self, where):
        '''Appends items from where to the where primitives to be used
           in DFS. Where primitives are used on specified interesting_values to 
           build conditional features and can take on aggregate or transform
           primitives.
           
           where: list of string values
        '''
        all_prims = ft.list_primitives()['name'].values
        for i in where:
            if i in all_prims:
                self.where_primitives.append(i)
            else:
                print(i, "is not in the available primitives")
        print("The where primitives have been added: ", 
              *self.where_primitives)

    def remove_primitive(self, prim, agg=False, trans=False, where=False):
        '''Removes the primitive specified, prim, from the list of agg_primitives,
           trans_primitives, or where_primitives if agg, trans, or where are True 
           respectively.
           
           prim: list of string values
        '''
        if agg:
            for i in prim:
                if i in self.agg_primitives:
                    self.agg_primitives.remove(i)
                    print(i, "has been removed")
                else:
                    print(i, "is not in the list of aggregate primitives")
        elif trans:
            for i in prim:
                if i in self.trans_primitives:
                    self.trans_primitives.remove(i)
                    print(i, "has been removed")
                else:
                    print(i, "is not in the list of transformative primitives")
    
        elif where:
            for i in prim:
                if i in self.where_primitives:
                    self.where_primitives.remove(i)
                    print(i, "has been removed")
                else:
                    print(i, "is not in the list of where primitives")
    
    def dfsWindow(self, target_entity, time_scope=None, training_window=None, 
                   cutoff_times=None, max_depth=1, chunk_size=None, n_jobs=1):
        '''Runs dfs on the target_entity and outputs a feature matrix with 
        features based on the training_window and time_scope relative to cutoff 
        times. If no training_window, time_scope, or cutoff_times are specified,
        regular dfs will run without using cutoff times.
           
        target_entity: str. Name of target_entity in entity set to run dfs on. 
        The index of the target_entity must match the instance_id column in the 
        cutoff_times table.
           
        time_scope: 'daily', 'weekly' or 'monthly'. Assumes 7 days in a week, 
        and 31 days in a month.
           
       training_window: list of integers that refer to the number of months or 
       weeks depending on the time_scope. Ex. [1, 2] for time_scope='monthly' 
       returns features based on the last month and last 2 months from the 
       cutoff date.
       
       cutoff_times: Pandas dataframe with instance_id, cutoff_dates, and 
       label (label is optional). Any columns after instance_id and cutoff_dates 
       will not be used for feature synthesis. The instance_id column must match 
       the index of the target entity. 
       
       max_depth: integer, defines how many levels of dfs to run. For example if 
       max_depth = 2 on a transactions table, features returned include avg. 
       transactions and avg. of avg. transactions.
       
       chunk_size: integer, float, None, or "cutoff time". Number of rows of 
       output feature matrix to calculate at time. If passed an integer greater 
       than 0, it will use that many rows per chunk. If passed a float value 
       between 0 and 1, sets the chunk size to that percentage of all instances. 
       If passed “cutoff time”, rows are split per cutoff time.
       
       n_jobs: integer. The number of parallel processes to use when creating
       the feature matrix.
        '''
        orig_window = training_window
        if (time_scope is None) or (training_window is None) or (cutoff_times is None):
            self.df, feature_defs = ft.dfs(entityset=self.es, 
                                          target_entity=target_entity, 
                                          agg_primitives = self.agg_primitives,
                                          trans_primitives = self.trans_primitives,
                                          where_primitives = self.where_primitives,
                                          max_depth = max_depth, features_only = False,
                                          verbose = 1, chunk_size = chunk_size,
                                          n_jobs = n_jobs)
        
        else:
            self.df, feature_defs = ft.dfs(entityset=self.es, 
                                          target_entity=target_entity, 
                                          cutoff_time = cutoff_times, 
                                          agg_primitives = self.agg_primitives,
                                          trans_primitives = self.trans_primitives,
                                          where_primitives = self.where_primitives,
                                          max_depth = max_depth, features_only = False,
                                          verbose = 1, chunk_size = chunk_size,
                                          n_jobs = n_jobs,
                                          cutoff_time_in_index = True)
            if time_scope == 'daily':
                training_window = [int(x) for x in orig_window]
                for i in range(len(training_window)):
                    feature_matrix = ft.calculate_feature_matrix(entityset=self.es,
                                             features=feature_defs,
                                             cutoff_time=cutoff_times, chunk_size = chunk_size,
                                             cutoff_time_in_index = True,
                                             n_jobs = n_jobs,
                                             training_window=ft.Timedelta(training_window[i], "d"))
                    
                    suffix = '_' + str(orig_window[i]) +'day'
                    feature_matrix = feature_matrix.add_suffix(suffix)
                    self.df = pd.concat([self.df, feature_matrix], axis=1, join='inner')
                    
            elif time_scope == 'monthly':
                training_window = [x*30 for x in orig_window]
                for i in range(len(training_window)):
                    feature_matrix = ft.calculate_feature_matrix(entityset=self.es,
                                             features=feature_defs,
                                             cutoff_time=cutoff_times, chunk_size = chunk_size,
                                             cutoff_time_in_index = True,
                                             n_jobs = n_jobs,
                                             training_window=ft.Timedelta(training_window[i], "d"))
                    
                    suffix = '_' + str(orig_window[i]) +'mos'
                    feature_matrix = feature_matrix.add_suffix(suffix)
                    self.df = pd.concat([self.df, feature_matrix], axis=1, join='inner')
            
            elif time_scope == 'weekly':
                training_window = [x*7 for x in orig_window]
                for i in range(len(training_window)):
                    feature_matrix, feature_defs = ft.dfs(entityset=self.es, 
                                          target_entity=target_entity, 
                                          cutoff_time = cutoff_times, 
                                          agg_primitives = self.agg_primitives,
                                          trans_primitives = self.trans_primitives,
                                          where_primitives = self.where_primitives,
                                          max_depth = max_depth, features_only = False,
                                          verbose = 1, chunk_size = chunk_size,  
                                          cutoff_time_in_index = True,
                                          n_jobs = n_jobs,
                                          training_window=ft.Timedelta(training_window[i], "d"))
                    
                    suffix = '_' + str(orig_window[i]) +'wks'
                    feature_matrix = feature_matrix.add_suffix(suffix)
                    self.df = pd.concat([self.df, feature_matrix], axis=1, join='inner')
                
            else:
                print("ERROR: time_scope entered is not one of the options.")
            
        drop_duplicates = DropDuplicate()
        self.df = drop_duplicates.fit_transform(self.df)
        
        for i in self.df.columns:
            self.feature_defs.append(i)

        return self.df
                
    def calc_avg_change(self, start, end, time_scope):
        '''Calculates the average rate of change from columns containing the 
        start value to columns containing the end end value of the specified
        time_scope. 
        
        For example, if the df contains a column called SUM(amt_paid)_6mos and 
        SUM(amt_paid)_3mos and the user inputs start = 3, end = 6, 
        time_scope = 'monthly', the average rate of change of the two columns 
        containing the  start and end timeframes are returned 
        (ie (SUM(amt_paid)_6mos - SUM(amt_paid)_3mos) / 3).

        start: integer from 1 to 12, references columns with the start month or 
        week for calculating average rate of change.

        end: integer from 1 to 12, references columns with the end month or week 
        for calculating average rate of change.

        time_scope: "daily", "monthly" or "weekly", defines the time scope for the 
        average rate of change.
        '''
        
        if end > start:
            diff = end - start
            start, end = str(start), str(end)
            low, high = [], []
            cols = sorted(list(self.df.columns))
           
            #Find all columns with start and end periods and store them in a list
            for i in cols:
                x = i.split('_')[-1]
                if time_scope == 'daily':
                    scope = 'day'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                elif time_scope == 'monthly':
                    scope = 'mos'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                elif time_scope == 'weekly':
                    scope = 'wks'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                else:
                    print("ERROR: time_scope can only be 'daily', 'weekly', or 'monthly'")
            
            low = sorted(low)
            high = sorted(high)
            for i in range(len(low)):
                x = low[i].split(')')
                col_name = ')'.join(x[:-1])+')_'+start+'-'+end+scope+'_avgchange'
                self.df[col_name] = (self.df[high[i]] - self.df[low[i]])/diff
                self.feature_defs.append(col_name)
                        
        else:
            print("ERROR: end is smaller than start.")
        return self.df

    def calc_diff(self, start, end, time_scope):
        '''Calculates the difference between columns containing the 
        start month or week to columns containing the end month or week. 
        For example, if the df contains a column called SUM(amt_paid)_6mos and 
        SUM(amt_paid)_3mos the difference of the two columns is returned 
        (ie (SUM(amt_paid)_6mos - SUM(amt_paid)_3mos) ).

        start: integer from 1 to 12, references columns with the start month or 
        week for calculating the difference.

        end: integer from 1 to 12, references columns with the end month or week 
        for calculating the difference.

        time_scope: "daily", "monthly" or "weekly", defines the time scope for the 
        difference calculated.
        '''
        if end > start:
            start, end = str(start), str(end)
            low, high = [], []
            cols = sorted(list(self.df.columns))
            #Find all columns with start and end periods and store them in a list
            for i in cols:
                x = i.split('_')[-1]
                if time_scope == 'daily':
                    scope = 'day'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                elif time_scope == 'monthly':
                    scope = 'mos'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                elif time_scope == 'weekly':
                    scope = 'wks'
                    if x[:-3] == end and x[-3:] == scope:
                        high.append(i)
                    elif x[:-3] == start and x[-3:] == scope:
                        low.append(i)
                else:
                    print("ERROR: time_scope can only be 'daily', 'weekly', or 'monthly'")
            
            low = sorted(low)
            high = sorted(high)
            for i in range(len(low)):
                x = low[i].split(')')
                col_name = ')'.join(x[:-1])+')_'+start+'-'+end+scope+'_diff'
                self.df[col_name] = (self.df[high[i]] - self.df[low[i]])
                self.feature_defs.append(col_name)
                
        else:
            print("ERROR: end is smaller than start.")
        return self.df
