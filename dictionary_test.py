import numpy as np

d = dict([
    ('methanotrophs',np.array([1,2,3])),

    ('acemethanogens',np.array([4,5,6]))
])

d['methanotrophs'][0] = 5

print(d['methanotrophs'])
