import gzip
import pickle
import sys

from ..utils.data_utils import get_file


def load_data(path='keras_mnistm.pkl.gz'):
    path = get_file(path, origin='https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz')

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')

    f.close()
    return data
