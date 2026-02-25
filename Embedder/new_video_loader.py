from pprint import pprint
import pickle
import tqdm

from torch.utils.data import Dataset
from Embedder.Embedder_config import config
from tqdm import tqdm

class Video_Loader(Dataset):
    def __init__(self, config):
        self.config = config
        self.videos = self.get_data()
        # self.vocab = self.get_vocab()
        pprint('NUMBER OF VIDEOS:' + str(len(self.videos)))

    def get_data(self):
        with open(self.config.DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        #
        pprint('VIDEO FILE SUCCESSFULLY LOADED')
        return data

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, workout_class = self.videos[idx]
        return video, workout_class

    def collate_fn(self, batch):
        videos, exercise_name = zip(*batch)

        return videos, exercise_name

if __name__ == '__main__':
    loader = Video_Loader(config)
    print('done')