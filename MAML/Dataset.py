
'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix, self.train_num = self.load_rating_file_as_matrix(path + "/train.csv")
        self.testRatings, self.test_num = self.load_rating_file_as_matrix(path + "/test.csv")
        print(self.trainMatrix.shape, self.testRatings.shape)
        self.textualfeatures,self.imagefeatures, self.meta_data = self.load_features(path)
        print(self.textualfeatures.shape, self.imagefeatures.shape)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_matrix(self, filename):



        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        print(num_users, num_items)
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item ,rating = int(row['userID']), int(row['itemID']) ,1.0
            if (rating > 0):
                mat[user, item] = 1.0
                num_total += 1
        return mat, num_total

    def load_features(self,data_path):
        import os
        from gensim.models.doc2vec import Doc2Vec
        # Prepare textual feture data.
        # doc2vec_model = Doc2Vec.load(os.path.join(data_path, 'doc2vecFile'))
        doc2vec_model = np.load(os.path.join(data_path, 'FeatureText_normal.npy'), allow_pickle=True)
        vis_vec = np.load(os.path.join(data_path, 'FeatureImage_normal.npy'), allow_pickle=True)
        meta_data = np.load(os.path.join(data_path, 'MetaData_normal.npy'), allow_pickle=True)
        # print('特征加载')
        # print(doc2vec_model.shape, vis_vec.shape)
        filename = data_path + '/train.csv'
        filename_test =  data_path + '/test.csv'
        df = pd.read_csv(filename, index_col=None, usecols=None)
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)
        num_items = 0
        asin_i_dic = {}
        index_asin = {}
        for index, row in df.iterrows():
             asin, i = row['asin'], int(row['itemID'])
             asin_i_dic[i] = asin
             index_asin[asin] = i
             num_items = max(num_items, i)
        # print('train', num_items)
        # train_asin_keys = index_asin.keys()
        # train_asin_index = index_asin.values()
        # num_items = 0
        # asin_i_dic = {}
        # index_asin = {}
        # df_test = df_test[df_test['itemID'] <= 11666]
        # df_test.to_csv(data_path + '/test.csv')
        #
        for index, row in df_test.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            index_asin[asin] = i
            num_items = max(num_items, i)
        print('train', num_items)
        # test_asin_keys = index_asin.keys()
        # test_asin_index = index_asin.values()
        # print(len(train_asin_keys), len(test_asin_keys))
        # exit()
        #
        # features = []
        # image_features = []
        # print(doc2vec_model)
        # for i in range(num_items+1):
        #     print(asin_i_dic[i])
        #     features.append(doc2vec_model[asin_i_dic[i]])
        #     image_features.append(vis_vec[asin_i_dic[i]])
        #
        # features = np.asarray(features,dtype=np.float32)
        # image_features = np.asarray(image_features, dtype=np.float32)
        # print(features.shape)
        # print(image_features.shape)
        # exit()
        # print('数据加载完毕')
        # features = features.reshape(-1, features.shape[2])
        doc2vec_model = doc2vec_model[:num_items+1]
        vis_vec = vis_vec[:num_items+1]
        meta_data = meta_data[:num_items+1]

        return doc2vec_model, vis_vec, meta_data