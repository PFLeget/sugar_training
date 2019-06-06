import pickle
import sys

def load_pickle(pickle_file):
    if sys.version_info[0] < 3:
        dico = pickle.load(open(pickle_file))
    else:
        dico = pickle.load(open(pickle_file, 'rb'), encoding='latin1')

    return dico

def write_pickle(dico, pickle_file):
    if sys.version_info[0] < 3:
        File = open(pickle_file, 'w')
    else:
        File = open(pickle_file, 'wb')
    pickle.dump(dico,File)
    File.close()
    return dico

