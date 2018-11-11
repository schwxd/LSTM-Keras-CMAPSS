import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# column names of CMAPSS Dataset
# CMAPSS数据集列名
columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

feature_columns = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 'cycle_norm']

class CMAPSSDataset():
    def __init__(self, fd_number, batch_size, sequence_length):
        super(CMAPSSDataset).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_data = None
        self.test_data = None
        self.train_data_encoding = None
        self.test_data_encoding = None
        
        # \s+ 匹配一个或多个空格
        data = pd.read_csv("C-MAPSS-Data\\train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        data.columns = columns
        
        # 计算该数据集包含的engine数目
        self.engine_size = data['id'].unique().max()
        
        # 计算每一行的剩余cycle
        rul = pd.DataFrame(data.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        data = data.merge(rul, on=['id'], how='left')
        data['RUL'] = data['max'] - data['cycle']
        #data.drop(['cycle', 'setting1', 'setting2', 'setting3'], axis=1, inplace=True)
        data.drop(['max'], axis=1, inplace=True)
        
        # 将id之外的列正规化
        self.std = StandardScaler()
        data['cycle_norm'] = data['cycle']
        cols_normalize = data.columns.difference(['id', 'cycle', 'RUL'])
        norm_data = pd.DataFrame(self.std.fit_transform(data[cols_normalize]), 
                                 columns=cols_normalize, index=data.index)
        join_data = data[data.columns.difference(cols_normalize)].join(norm_data)
        self.train_data = join_data.reindex(columns=data.columns)
        
        # 读取测试数据集并执行相同操作
        # 测试集engine完整的rul还需要包括RUL_FD00x.txt文件中的部分
        test_data = pd.read_csv("C-MAPSS-Data\\test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        test_data.columns = columns
        truth_data = pd.read_csv("C-MAPSS-Data\\RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        truth_data.columns = ['truth']
        truth_data['id'] = truth_data.index + 1
        
        test_rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
        test_rul.columns = ['id', 'elapsed']
        test_rul = test_rul.merge(truth_data, on=['id'], how='left')
        test_rul['max'] = test_rul['elapsed'] + test_rul['truth']
        
        test_data = test_data.merge(test_rul, on=['id'], how='left')
        test_data['RUL'] = test_data['max'] - test_data['cycle']
        test_data.drop(['max'], axis=1, inplace=True)
        
        test_data['cycle_norm'] = test_data['cycle']
        norm_test_data = pd.DataFrame(self.std.fit_transform(test_data[cols_normalize]), 
                                 columns=cols_normalize, index=test_data.index)
        join_test_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
        self.test_data = join_test_data.reindex(columns=test_data.columns)
     
    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
        
    def get_feature_slice(self, input_data):
        # Reshape the data to (samples, time steps, features)
        def reshapeFeatures(input, columns, sequence_length):
            data = input[columns].values
            num_elements = data.shape[0]
            #print(num_elements)
            for start, stop in zip(range(0, num_elements-sequence_length), range(sequence_length, num_elements)):
                yield(data[start:stop, :])
                
        feature_list = [list(reshapeFeatures(input_data[input_data['id'] == i], feature_columns, self.sequence_length)) 
                        for i in range(1, self.engine_size + 1) if len(input_data[input_data['id']  == i]) > self.sequence_length]
        ##for i in range(len(feature_list)):
        ##    print(np.array(feature_list[i]).shape)
        feature_array = np.concatenate(list(feature_list), axis=0).astype(np.float32)

        length = len(feature_array) // self.batch_size
        return feature_array[:length*self.batch_size]
    
    #
    # get the engine id of dataset
    # In VAE the encoded dataset need to be reshape again (using sliding window within each engine)
    # so the engine id need to be reserved
    # #
    def get_engine_id(self, input_data):
        def reshapeLabels(input, sequence_length, columns=['id']):
            data = input[columns].values
            num_elements = data.shape[0]
            return(data[sequence_length:num_elements, :])
                
        label_list = [reshapeLabels(input_data[input_data['id'] == i], self.sequence_length) 
              for i in range(1, self.engine_size+1)]
        label_array = np.concatenate(label_list).astype(np.int8)
        length = len(label_array) // self.batch_size
        return label_array[:length*self.batch_size]
        
    def get_label_slice(self, input_data):
        def reshapeLabels(input, sequence_length, columns=['RUL']):
            data = input[columns].values
            num_elements = data.shape[0]
            return(data[sequence_length:num_elements, :])
                
        label_list = [reshapeLabels(input_data[input_data['id'] == i], self.sequence_length) 
              for i in range(1, self.engine_size+1)]
        label_array = np.concatenate(label_list).astype(np.float32)
        length = len(label_array) // self.batch_size
        return label_array[:length*self.batch_size]
   
    # 每个engine只取最后一个sequence_length（如果该engine的数据条目数大于sequence_length的话，否则就舍弃）
    # 用于最后的evaluation
    def get_last_data_slice(self, input_data):
        num_engine = input_data['id'].unique().max()
        test_feature_list = [input_data[input_data['id'] == i][feature_columns].values[-self.sequence_length:] 
                             for i in range(1, num_engine+1) if len(input_data[input_data['id']  == i]) >= self.sequence_length]
        test_feature_array = np.asarray(test_feature_list).astype(np.float32)
        length_test = len(test_feature_array) // self.batch_size

        test_label_list = [input_data[input_data['id'] == i]['RUL'].values[-1:] 
                           for i in range(1, num_engine+1) if len(input_data[input_data['id'] == i]) >= self.sequence_length]
        test_label_array = np.asarray(test_label_list).astype(np.float32)
        length_label = len(test_label_array) // self.batch_size

        return test_feature_array[:length_test*self.batch_size], test_label_array[:length_label*self.batch_size]

    # 
    def set_test_data_encoding(self, test_data_encoding):
        self.test_data_encoding = test_data_encoding

    def set_train_data_encoding(self, train_data_encoding):
        self.train_data_encoding = train_data_encoding

if __name__ == "__main__":
    datasets = CMAPSSDataset(fd_number='1', batch_size=10, sequence_length=50)

    train_data = datasets.get_train_data()
    train_feature_slice = datasets.get_feature_slice(train_data)
    train_label_slice = datasets.get_label_slice(train_data)
    train_engine_id = datasets.get_engine_id(train_data)
    print("train_data.shape: {}".format(train_data.shape))
    print("train_feature_slice.shape: {}".format(train_feature_slice.shape))
    print("train_label_slice.shape: {}".format(train_label_slice.shape))
    print("train_engine_id.shape: {}".format(train_engine_id.shape))

    test_data = datasets.get_test_data()
    print("test_data.shape: {}".format(test_data.shape))
    test_feature_slice = datasets.get_feature_slice(test_data)
    test_label_slice = datasets.get_label_slice(test_data)
    test_engine_id = datasets.get_engine_id(test_data)
    print("test_feature_slice.shape: {}".format(test_feature_slice.shape))
    print("test_label_slice.shape: {}".format(test_label_slice.shape))
    print("test_engine_id.shape: {}".format(test_engine_id.shape))

    """
    np.savetxt('train_engine_id.txt', train_engine_id, fmt='%d')
    np.savetxt('test_engine_id.txt', test_engine_id, fmt='%d')
    data_batch = datasets.get_train_dataset_batch()

    print(type(data_batch))
    print(np.array(data_batch).shape)
    data_batch_tensor = tf.convert_to_tensor(data_batch)
    print(data_batch_tensor.shape)
    """