from FeatureSelection import *


def test_duplicate():
    test_data = pd.DataFrame({'a': [None]*9 + [1],
                              'b': np.ones(10),
                              'c': np.ones(10),
                              'd': 'cat'})

    dropduplicate = DropDuplicate()
    
    result = dropduplicate.fit_transform(test_data)
    
    assert result.equals(pd.DataFrame({'a': [None]*9 + [1],
                                       'b': np.ones(10),
                                       'd': 'cat' }))

def test_missing():
  test_data = pd.DataFrame({'a': [None]*9 + [1],
                            'b': np.ones(10),
                           'c': range(10),
                           'd': 'cat'})

  dropmissing = DropMissing(threshold=0.9)

  result = dropmissing.fit_transform(test_data)

  assert result.equals(pd.DataFrame({'b': np.ones(10),
                                    'c': range(10),
                                    'd': 'cat'})) 

def test_highcorr():
 drophighcorr = DropHighCorr(threshold=0.9)
 test_data = pd.DataFrame({'a': np.array(range(10)), 
                           'b': np.array(range(10)) + 0.05,
                           'c': np.ones(10) 
                           })

 result = drophighcorr.fit_transform(test_data)

 assert result.equals(pd.DataFrame({'a': np.array(range(10)), 
                                    'c': np.ones(10) 
                                    }))

def test_zerocov():
 dropzerocov = DropZeroCov()
 test_data = pd.DataFrame({'a': np.array(range(10)), 
                           'b': np.array(range(10)) + 0.05,
                           'c': np.ones(10) 
                          })
 result = dropzerocov.fit_transform(test_data)
 assert result.equals(pd.DataFrame({'a': np.array(range(10)), 
                                    'b': np.array(range(10)) + 0.05,
                                    }))

