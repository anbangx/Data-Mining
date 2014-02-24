__author__ = 'anbangx'

import cPickle as pickle

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'r') as f:
        return pickle.load(f)

# dict = {'a': [100, 200], 'b': [300, 50]}
# save_obj(dict, 'dict')
# dict = load_obj('adjust_depth')
# print dict

for i in range(5, 10, 5):
    print str(i)