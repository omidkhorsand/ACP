import pandas as pd 
import numpy as np
import featuretools as ft
import featuretools.variable_types as vtypes
from FeatureExtraction import FeatureExtraction
from pandas.util.testing import assert_frame_equal

#First we create the 3 datasets to use in Featuretools
#These will be used to evaluate the monthly feature calculations on FeatureExtraction
#Later, the trans dates will be altered to test the weekly feature calculations
members = [[1,'f','2016-06-18'],
           [2,'f','2016-06-19'],
           [3,'m','2016-06-20']]

members = pd.DataFrame(members, columns =['id','gender','registration_date'])
members['registration_date'] = members['registration_date'].astype('datetime64[ns]')

trans = [[1,30,2,'2017-01-01','2017-02-28'],
        [1,60,4,'2017-02-25','2017-04-03'],
        [1,90,6,'2017-03-15','2017-03-16'],
        [2,250,8,'2017-01-01','2017-02-28'],
        [2,50,10,'2017-02-01','2017-03-15'],
        [2,100,2,'2017-04-30','2017-05-20'],
        [3,1000,3,'2017-07-01','2017-08-01'],
        [3,500,4,'2017-08-01','2017-09-01'],
        [3,250,5,'2017-10-15','2017-11-15']]

trans = pd.DataFrame(trans, columns = ['id','amt_paid', 'qty',
                                       'transaction_date',
                                       'membership_expire_date'])
trans[['transaction_date', 'membership_expire_date']] = \
trans[['transaction_date', 'membership_expire_date']].astype('datetime64[ns]')

cutoff_times = [[1, '2017-03-15', 0],
       [2, '2017-02-01', 1],
       [3, '2017-09-01', 1]]

cutoff_times = pd.DataFrame(cutoff_times, columns = ['instance_id', 'cutoff_time','label'])
cutoff_times['cutoff_time'] = cutoff_times['cutoff_time'].astype('datetime64[ns]')

#Now we create the Entity Sets and define the relationships
es = ft.EntitySet(id = 'customers')
es.entity_from_dataframe(entity_id='members', dataframe=members,
                         index = 'id', time_index = 'registration_date', 
                         variable_types = {'gender': vtypes.Categorical})


#Create entity from transactions
es.entity_from_dataframe(entity_id='transactions', dataframe=trans, 
                         index = 'transactions_index', make_index = True,
                         time_index = 'transaction_date')

r_member_transactions = ft.Relationship(es['members']['id'], 
                                        es['transactions']['id'])
es.add_relationships([r_member_transactions])
es.add_last_time_indexes()

#Creating datasets to check against
monthly = [['f', 300, 18, 150, 9.0, 6, 1, 50.0, 10.0, 50.0, 10.0, 300, 18, 150,
        9],
       ['f', 180, 12, 60, 4.0, 6, 0, 150.0, 10.0, 75.0, 5.0, 150, 10, 75, 5],
       ['m', 1500, 7, 750, 3.5, 6, 1, 0.0, 0.0, np.nan, np.nan, 500, 4, 500, 4]]

ind = pd.DataFrame(data=[[2, '2017-02-01'], [1, '2017-03-15'],  
                         [3, '2017-09-01']], columns=['id','time'])
ind['time'] = ind['time'].astype('datetime64[ns]')
ind = pd.MultiIndex.from_frame(ind)

monthly = pd.DataFrame(monthly, columns=['gender',
                                 'SUM(transactions.amt_paid)',
                                 'SUM(transactions.qty)',
                                 'MEAN(transactions.amt_paid)',
                                 'MEAN(transactions.qty)',
                                 'MONTH(registration_date)',
                                 'label',
                                 'SUM(transactions.amt_paid)_1mos',
                                 'SUM(transactions.qty)_1mos',
                                 'MEAN(transactions.amt_paid)_1mos',
                                 'MEAN(transactions.qty)_1mos',
                                 'SUM(transactions.amt_paid)_2mos',
                                 'SUM(transactions.qty)_2mos',
                                 'MEAN(transactions.amt_paid)_2mos',
                                 'MEAN(transactions.qty)_2mos'], index=ind )

monthly_feats = ['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'MONTH(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1mos',
                 'SUM(transactions.qty)_1mos',
                 'MEAN(transactions.amt_paid)_1mos',
                 'MEAN(transactions.qty)_1mos',
                 'SUM(transactions.amt_paid)_2mos',
                 'SUM(transactions.qty)_2mos',
                 'MEAN(transactions.amt_paid)_2mos',
                 'MEAN(transactions.qty)_2mos']

monthly_avg_chg = [['f', 300, 18, 150, 9.0, 6, 1, 50.0, 10.0, 50.0, 10.0, 300, 18, 150,
        9, 100.0, -1.0, 250.0, 8.0],
       ['f', 180, 12, 60, 4.0, 6, 0, 150.0, 10.0, 75.0, 5.0, 150, 10, 75,
        5, 0.0, 0.0, 0.0, 0.0],
       ['m', 1500, 7, 750, 3.5, 6, 1, 0.0, 0.0, np.nan, np.nan, 500, 4, 500, 4,
        np.nan, np.nan, 500.0, 4.0]]

monthly_avg_chg = pd.DataFrame(monthly_avg_chg, 
        columns=['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'MONTH(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1mos',
                 'SUM(transactions.qty)_1mos',
                 'MEAN(transactions.amt_paid)_1mos',
                 'MEAN(transactions.qty)_1mos',
                 'SUM(transactions.amt_paid)_2mos',
                 'SUM(transactions.qty)_2mos',
                 'MEAN(transactions.amt_paid)_2mos',
                 'MEAN(transactions.qty)_2mos',
                 'MEAN(transactions.amt_paid)_1-2mos_avgchange',
                 'MEAN(transactions.qty)_1-2mos_avgchange',
                 'SUM(transactions.amt_paid)_1-2mos_avgchange',
                 'SUM(transactions.qty)_1-2mos_avgchange'], index=ind )

monthly_feats_avg_chg = ['gender',
                         'SUM(transactions.amt_paid)',
                         'SUM(transactions.qty)',
                         'MEAN(transactions.amt_paid)',
                         'MEAN(transactions.qty)',
                         'MONTH(registration_date)',
                         'label',
                         'SUM(transactions.amt_paid)_1mos',
                         'SUM(transactions.qty)_1mos',
                         'MEAN(transactions.amt_paid)_1mos',
                         'MEAN(transactions.qty)_1mos',
                         'SUM(transactions.amt_paid)_2mos',
                         'SUM(transactions.qty)_2mos',
                         'MEAN(transactions.amt_paid)_2mos',
                         'MEAN(transactions.qty)_2mos',
                         'MEAN(transactions.amt_paid)_1-2mos_avgchange',
                         'MEAN(transactions.qty)_1-2mos_avgchange',
                         'SUM(transactions.amt_paid)_1-2mos_avgchange',
                         'SUM(transactions.qty)_1-2mos_avgchange']

monthly_diff = [['f', 300, 18, 150, 9.0, 6, 1, 50.0, 10.0, 50.0, 10.0, 300, 18, 150,
        9, 100.0, -1.0, 250.0, 8.0],
       ['f', 180, 12, 60, 4.0, 6, 0, 150.0, 10.0, 75.0, 5.0, 150, 10, 75,
        5, 0.0, 0.0, 0.0, 0.0],
       ['m', 1500, 7, 750, 3.5, 6, 1, 0.0, 0.0, np.nan, np.nan, 500, 4, 500, 4,
        np.nan, np.nan, 500.0, 4.0]]

monthly_diff = pd.DataFrame(monthly_diff, 
        columns=['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'MONTH(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1mos',
                 'SUM(transactions.qty)_1mos',
                 'MEAN(transactions.amt_paid)_1mos',
                 'MEAN(transactions.qty)_1mos',
                 'SUM(transactions.amt_paid)_2mos',
                 'SUM(transactions.qty)_2mos',
                 'MEAN(transactions.amt_paid)_2mos',
                 'MEAN(transactions.qty)_2mos',
                 'MEAN(transactions.amt_paid)_1-2mos_diff',
                 'MEAN(transactions.qty)_1-2mos_diff',
                 'SUM(transactions.amt_paid)_1-2mos_diff',
                 'SUM(transactions.qty)_1-2mos_diff'], index=ind )

monthly_feats_diff = ['gender',
                     'SUM(transactions.amt_paid)',
                     'SUM(transactions.qty)',
                     'MEAN(transactions.amt_paid)',
                     'MEAN(transactions.qty)',
                     'MONTH(registration_date)',
                     'label',
                     'SUM(transactions.amt_paid)_1mos',
                     'SUM(transactions.qty)_1mos',
                     'MEAN(transactions.amt_paid)_1mos',
                     'MEAN(transactions.qty)_1mos',
                     'SUM(transactions.amt_paid)_2mos',
                     'SUM(transactions.qty)_2mos',
                     'MEAN(transactions.amt_paid)_2mos',
                     'MEAN(transactions.qty)_2mos',
                     'MEAN(transactions.amt_paid)_1-2mos_diff',
                     'MEAN(transactions.qty)_1-2mos_diff',
                     'SUM(transactions.amt_paid)_1-2mos_diff',
                     'SUM(transactions.qty)_1-2mos_diff']

#For the weekly dataset we simply adjust the trans dates to shorter periods to 
#show changes by week
trans = [[1,30,2,'2017-03-01','2017-03-02'],
        [1,60,4,'2017-03-08','2017-03-09'],
        [1,90,6,'2017-03-15','2017-03-16'],
        [2,250,8,'2017-01-31','2017-02-28'],
        [2,50,10,'2017-02-01','2017-03-15'],
        [2,100,2,'2017-04-30','2017-05-20'],
        [3,1000,3,'2017-07-01','2017-08-01'],
        [3,500,4,'2017-08-25','2017-09-01'],
        [3,250,5,'2017-10-15','2017-11-15']]

trans = pd.DataFrame(trans, columns = ['id','amt_paid', 'qty',
                                       'transaction_date',
                                       'membership_expire_date'])
trans[['transaction_date', 'membership_expire_date']] = \
trans[['transaction_date', 'membership_expire_date']].astype('datetime64[ns]')

#Now we create the Entity Sets and define the relationships
esw = ft.EntitySet(id = 'customers')
esw.entity_from_dataframe(entity_id='members', dataframe=members,
                         index = 'id', time_index = 'registration_date', 
                         variable_types = {'gender': vtypes.Categorical})


#Create entity from transactions
esw.entity_from_dataframe(entity_id='transactions', dataframe=trans, 
                         index = 'transactions_index', make_index = True,
                         time_index = 'transaction_date')

r_member_transactions = ft.Relationship(esw['members']['id'], 
                                        esw['transactions']['id'])
esw.add_relationships([r_member_transactions])
esw.add_last_time_indexes()

weekly = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4]]

weekly = pd.DataFrame(weekly, 
        columns=['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'WEEK(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1wks',
                 'SUM(transactions.qty)_1wks',
                 'MEAN(transactions.amt_paid)_1wks',
                 'MEAN(transactions.qty)_1wks',
                 'SUM(transactions.amt_paid)_2wks',
                 'SUM(transactions.qty)_2wks',
                 'MEAN(transactions.amt_paid)_2wks',
                 'MEAN(transactions.qty)_2wks'], index=ind )


weekly_feats = ['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'WEEK(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1wks',
                 'SUM(transactions.qty)_1wks',
                 'MEAN(transactions.amt_paid)_1wks',
                 'MEAN(transactions.qty)_1wks',
                 'SUM(transactions.amt_paid)_2wks',
                 'SUM(transactions.qty)_2wks',
                 'MEAN(transactions.amt_paid)_2wks',
                 'MEAN(transactions.qty)_2wks']

weekly_avg_chg = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9,
        0.0, 0.0, 0.0, 0.0],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4, -7.5,
        -0.5, 15.0, 1.0],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4, 0.0,
        0.0, 0.0, 0.0]]

weekly_avg_chg = pd.DataFrame(weekly_avg_chg, 
        columns=['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'WEEK(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1wks',
                 'SUM(transactions.qty)_1wks',
                 'MEAN(transactions.amt_paid)_1wks',
                 'MEAN(transactions.qty)_1wks',
                 'SUM(transactions.amt_paid)_3wks',
                 'SUM(transactions.qty)_3wks',
                 'MEAN(transactions.amt_paid)_3wks',
                 'MEAN(transactions.qty)_3wks',
                 'MEAN(transactions.amt_paid)_1-3wks_avgchange',
                 'MEAN(transactions.qty)_1-3wks_avgchange',
                 'SUM(transactions.amt_paid)_1-3wks_avgchange',
                 'SUM(transactions.qty)_1-3wks_avgchange'], index=ind )

weekly_feats_avg_chg = ['gender',
                         'SUM(transactions.amt_paid)',
                         'SUM(transactions.qty)',
                         'MEAN(transactions.amt_paid)',
                         'MEAN(transactions.qty)',
                         'WEEK(registration_date)',
                         'label',
                         'SUM(transactions.amt_paid)_1wks',
                         'SUM(transactions.qty)_1wks',
                         'MEAN(transactions.amt_paid)_1wks',
                         'MEAN(transactions.qty)_1wks',
                         'SUM(transactions.amt_paid)_3wks',
                         'SUM(transactions.qty)_3wks',
                         'MEAN(transactions.amt_paid)_3wks',
                         'MEAN(transactions.qty)_3wks',
                         'MEAN(transactions.amt_paid)_1-3wks_avgchange',
                         'MEAN(transactions.qty)_1-3wks_avgchange',
                         'SUM(transactions.amt_paid)_1-3wks_avgchange',
                         'SUM(transactions.qty)_1-3wks_avgchange']

weekly_diff = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9, 0,
        0, 0, 0],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4, -15,
        -1, 30, 2],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4, 0,
        0, 0, 0]]

weekly_diff = pd.DataFrame(weekly_diff, 
        columns=['gender',
                 'SUM(transactions.amt_paid)',
                 'SUM(transactions.qty)',
                 'MEAN(transactions.amt_paid)',
                 'MEAN(transactions.qty)',
                 'WEEK(registration_date)',
                 'label',
                 'SUM(transactions.amt_paid)_1wks',
                 'SUM(transactions.qty)_1wks',
                 'MEAN(transactions.amt_paid)_1wks',
                 'MEAN(transactions.qty)_1wks',
                 'SUM(transactions.amt_paid)_3wks',
                 'SUM(transactions.qty)_3wks',
                 'MEAN(transactions.amt_paid)_3wks',
                 'MEAN(transactions.qty)_3wks',
                 'MEAN(transactions.amt_paid)_1-3wks_diff',
                 'MEAN(transactions.qty)_1-3wks_diff',
                 'SUM(transactions.amt_paid)_1-3wks_diff',
                 'SUM(transactions.qty)_1-3wks_diff'], index=ind )

weekly_feats_diff = ['gender',
                     'SUM(transactions.amt_paid)',
                     'SUM(transactions.qty)',
                     'MEAN(transactions.amt_paid)',
                     'MEAN(transactions.qty)',
                     'WEEK(registration_date)',
                     'label',
                     'SUM(transactions.amt_paid)_1wks',
                     'SUM(transactions.qty)_1wks',
                     'MEAN(transactions.amt_paid)_1wks',
                     'MEAN(transactions.qty)_1wks',
                     'SUM(transactions.amt_paid)_3wks',
                     'SUM(transactions.qty)_3wks',
                     'MEAN(transactions.amt_paid)_3wks',
                     'MEAN(transactions.qty)_3wks',
                     'MEAN(transactions.amt_paid)_1-3wks_diff',
                     'MEAN(transactions.qty)_1-3wks_diff',
                     'SUM(transactions.amt_paid)_1-3wks_diff',
                     'SUM(transactions.qty)_1-3wks_diff']

daily = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4]]

daily = pd.DataFrame(daily,
                     columns=['gender',
                             'SUM(transactions.amt_paid)',
                             'SUM(transactions.qty)',
                             'MEAN(transactions.amt_paid)',
                             'MEAN(transactions.qty)',
                             'WEEK(registration_date)',
                             'label',
                             'SUM(transactions.amt_paid)_7day',
                             'SUM(transactions.qty)_7day',
                             'MEAN(transactions.amt_paid)_7day',
                             'MEAN(transactions.qty)_7day',
                             'SUM(transactions.amt_paid)_14day',
                             'SUM(transactions.qty)_14day',
                             'MEAN(transactions.amt_paid)_14day',
                             'MEAN(transactions.qty)_14day'], index=ind)

daily_feats = ['gender',
             'SUM(transactions.amt_paid)',
             'SUM(transactions.qty)',
             'MEAN(transactions.amt_paid)',
             'MEAN(transactions.qty)',
             'WEEK(registration_date)',
             'label',
             'SUM(transactions.amt_paid)_7day',
             'SUM(transactions.qty)_7day',
             'MEAN(transactions.amt_paid)_7day',
             'MEAN(transactions.qty)_7day',
             'SUM(transactions.amt_paid)_14day',
             'SUM(transactions.qty)_14day',
             'MEAN(transactions.amt_paid)_14day',
             'MEAN(transactions.qty)_14day']

daily_avg_chg = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9,
        0.0, 0.0, 0.0, 0.0],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4,
        -2.142857142857143, -0.14285714285714285, 4.285714285714286,
        0.2857142857142857],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4, 0.0,
        0.0, 0.0, 0.0]]

daily_avg_chg = pd.DataFrame(daily_avg_chg,
                             columns=['gender',
                         'SUM(transactions.amt_paid)',
                         'SUM(transactions.qty)',
                         'MEAN(transactions.amt_paid)',
                         'MEAN(transactions.qty)',
                         'WEEK(registration_date)',
                         'label',
                         'SUM(transactions.amt_paid)_7day',
                         'SUM(transactions.qty)_7day',
                         'MEAN(transactions.amt_paid)_7day',
                         'MEAN(transactions.qty)_7day',
                         'SUM(transactions.amt_paid)_14day',
                         'SUM(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_14day',
                         'MEAN(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_7-14day_avgchange',
                         'MEAN(transactions.qty)_7-14day_avgchange',
                         'SUM(transactions.amt_paid)_7-14day_avgchange',
                         'SUM(transactions.qty)_7-14day_avgchange'], index=ind)

daily_feats_avg_chg = ['gender',
                         'SUM(transactions.amt_paid)',
                         'SUM(transactions.qty)',
                         'MEAN(transactions.amt_paid)',
                         'MEAN(transactions.qty)',
                         'WEEK(registration_date)',
                         'label',
                         'SUM(transactions.amt_paid)_7day',
                         'SUM(transactions.qty)_7day',
                         'MEAN(transactions.amt_paid)_7day',
                         'MEAN(transactions.qty)_7day',
                         'SUM(transactions.amt_paid)_14day',
                         'SUM(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_14day',
                         'MEAN(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_7-14day_avgchange',
                         'MEAN(transactions.qty)_7-14day_avgchange',
                         'SUM(transactions.amt_paid)_7-14day_avgchange',
                         'SUM(transactions.qty)_7-14day_avgchange']

daily_diff = [['f', 300, 18, 150, 9.0, 24, 1, 300, 18, 150, 9, 300, 18, 150, 9, 0,
        0, 0, 0],
       ['f', 180, 12, 60, 4.0, 24, 0, 150, 10, 75, 5, 180, 12, 60, 4, -15,
        -1, 30, 2],
       ['m', 1500, 7, 750, 3.5, 25, 1, 500, 4, 500, 4, 500, 4, 500, 4, 0,
        0, 0, 0]]

daily_diff = pd.DataFrame(daily_diff,
                          columns=['gender',
                         'SUM(transactions.amt_paid)',
                         'SUM(transactions.qty)',
                         'MEAN(transactions.amt_paid)',
                         'MEAN(transactions.qty)',
                         'WEEK(registration_date)',
                         'label',
                         'SUM(transactions.amt_paid)_7day',
                         'SUM(transactions.qty)_7day',
                         'MEAN(transactions.amt_paid)_7day',
                         'MEAN(transactions.qty)_7day',
                         'SUM(transactions.amt_paid)_14day',
                         'SUM(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_14day',
                         'MEAN(transactions.qty)_14day',
                         'MEAN(transactions.amt_paid)_7-14day_diff',
                         'MEAN(transactions.qty)_7-14day_diff',
                         'SUM(transactions.amt_paid)_7-14day_diff',
                         'SUM(transactions.qty)_7-14day_diff'],
                                                    index=ind)

daily_feats_diff = ['gender',
                     'SUM(transactions.amt_paid)',
                     'SUM(transactions.qty)',
                     'MEAN(transactions.amt_paid)',
                     'MEAN(transactions.qty)',
                     'WEEK(registration_date)',
                     'label',
                     'SUM(transactions.amt_paid)_7day',
                     'SUM(transactions.qty)_7day',
                     'MEAN(transactions.amt_paid)_7day',
                     'MEAN(transactions.qty)_7day',
                     'SUM(transactions.amt_paid)_14day',
                     'SUM(transactions.qty)_14day',
                     'MEAN(transactions.amt_paid)_14day',
                     'MEAN(transactions.qty)_14day',
                     'MEAN(transactions.amt_paid)_7-14day_diff',
                     'MEAN(transactions.qty)_7-14day_diff',
                     'SUM(transactions.amt_paid)_7-14day_diff',
                     'SUM(transactions.qty)_7-14day_diff']

#Now we start testing
#First we will test with timescope = 'monthly'
def test_add_agg_primitives():
    '''Test for adding aggregate primitives'''
    m = FeatureExtraction(es)
    m.add_agg_primitives(['sum','mean'])
    assert set(m.agg_primitives) == set(['sum','mean'])

def test_add_trans_primitives():
    '''Test for adding transformative primitives'''
    m = FeatureExtraction(es)
    m.add_trans_primitives(['week','day'])
    assert set(m.trans_primitives) == set(['week','day'])
    
def test_remove_primitive():
    '''Test for removing primitives'''
    m = FeatureExtraction(es)
    m.add_agg_primitives(['count'])
    m.add_trans_primitives(['month'])
    m.add_where_primitives(['median'])
    m.remove_primitive(['count'], agg=True)
    m.remove_primitive(['month'], trans=True)
    m.remove_primitive(['median'], where=True)
    assert m.agg_primitives == [] 
    assert m.trans_primitives == []
    assert m.where_primitives == []

def test_monthly():
    '''Test for dfsWindow with monthly timescope'''
    m = FeatureExtraction(es)
    m.add_agg_primitives(['sum', 'mean'])
    m.add_trans_primitives(['month'])
    training_window = [1,2]
    m.dfsWindow('members', 'monthly', training_window,   
                       cutoff_times, max_depth=1)
    assert set(m.feature_defs) == set(monthly_feats)
    assert_frame_equal(m.df,monthly)
    
def test_monthly_avg_chg():
    '''Test for calc_avg_chg on dfsWindow monthly timescope output'''
    m = FeatureExtraction(es)
    m.add_agg_primitives(['sum', 'mean'])
    m.add_trans_primitives(['month'])
    training_window = [1,2]
    m.dfsWindow('members', 'monthly', training_window,   
                       cutoff_times, max_depth=1)
    m.calc_avg_change(1,2,'monthly')
    assert set(m.feature_defs) == set(monthly_feats_avg_chg)
    assert_frame_equal(m.df,monthly_avg_chg)

def test_monthly_diff():
    '''Test for calc_diff on dfsWindow monthly timescope output'''
    m = FeatureExtraction(es)
    m.add_agg_primitives(['sum', 'mean'])
    m.add_trans_primitives(['month'])
    training_window = [1,2]
    m.dfsWindow('members', 'monthly', training_window,   
                       cutoff_times, max_depth=1)
    m.calc_diff(1,2,'monthly')
    assert set(m.feature_defs) == set(monthly_feats_diff)
    assert_frame_equal(m.df,monthly_diff)

#Now we test for timescope = 'weekly'
def test_weekly():
    '''Test for dfsWindow with weekly timescope'''
    w = FeatureExtraction(esw)
    w.add_agg_primitives(['sum', 'mean'])
    w.add_trans_primitives(['week'])
    training_window = [1,2]
    w.dfsWindow('members', 'weekly', training_window, cutoff_times, 
                      max_depth=1)
    assert set(w.feature_defs) == set(weekly_feats)
    assert_frame_equal(w.df,weekly)

def test_weekly_avg_chg():
    '''Test for calc_avg_chg on dfsWindow weekly timescope output'''
    w = FeatureExtraction(esw)
    w.add_agg_primitives(['sum', 'mean'])
    w.add_trans_primitives(['week'])
    training_window = [1,3]
    w.dfsWindow('members', 'weekly', training_window, cutoff_times, 
                      max_depth=1)
    w.calc_avg_change(1,3,'weekly')
    assert set(w.feature_defs) == set(weekly_feats_avg_chg)
    assert_frame_equal(w.df,weekly_avg_chg)
    
def test_weekly_diff():
    '''Test for calc_diff on dfsWindow weekly timescope output'''
    w = FeatureExtraction(esw)
    w.add_agg_primitives(['sum', 'mean'])
    w.add_trans_primitives(['week'])
    training_window = [1,3]
    w.dfsWindow('members', 'weekly', training_window, cutoff_times, 
                      max_depth=1)
    w.calc_diff(1,3,'weekly')
    assert set(w.feature_defs) == set(weekly_feats_diff)
    assert_frame_equal(w.df,weekly_diff)
    
#For timescope = 'daily' we set the days to 7 and 14
def test_daily():
    '''Test for dfsWindow with daily timescope'''
    d = FeatureExtraction(esw)
    d.add_agg_primitives(['sum', 'mean'])
    d.add_trans_primitives(['week'])
    training_window = [7,14]
    d.dfsWindow('members', 'daily', training_window, cutoff_times, 
                      max_depth=1)
    assert set(d.feature_defs) == set(daily_feats)
    assert_frame_equal(d.df,daily)

def test_daily_avg_chg():
    '''Test for calc_avg_chg on dfsWindow daily timescope output'''
    d = FeatureExtraction(esw)
    d.add_agg_primitives(['sum', 'mean'])
    d.add_trans_primitives(['week'])
    training_window = [7,14]
    d.dfsWindow('members', 'daily', training_window, cutoff_times, 
                      max_depth=1)
    d.calc_avg_change(7,14,'daily')
    assert set(d.feature_defs) == set(daily_feats_avg_chg)
    assert_frame_equal(d.df,daily_avg_chg)
    
def test_daily_diff():
    '''Test for calc_diff on dfsWindow daily timescope output'''
    d = FeatureExtraction(esw)
    d.add_agg_primitives(['sum', 'mean'])
    d.add_trans_primitives(['week'])
    training_window = [7,14]
    d.dfsWindow('members', 'daily', training_window, cutoff_times, 
                      max_depth=1)
    d.calc_diff(7,14,'daily')
    assert set(d.feature_defs) == set(daily_feats_diff)
    assert_frame_equal(d.df,daily_diff)

