import json
import pickle

'''
    [1] data_path = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_train.json'
    workout -> view -> num_video -> frame -> coord
    
    
    [2] data_path = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/train.pkl'
    data[n] means a video
    len(data[n]) = 2 -> data[0][0] means a frame, data[0][1] means a coord
    data[0][0].keys(): dict_keys(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
    data[0][0]['0']['Left Shoulder']: array([ 0.49635416,  0.537963  ,  2.        , 22.        ], dtype=float32)
    data[0][1]: 22 -> exercise name token
'''

data_path = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/train_contained_condition.json'

if data_path.split('/')[-1].split('.')[-1] == 'json':
    with open(data_path, 'r') as f:
        data = json.load(f)
        print()

elif data_path.split('/')[-1].split('.')[-1] == 'pkl':
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        print()


