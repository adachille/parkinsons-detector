import numpy as np

def read_csv_into_numpy(dataset):
    content = open(dataset).readlines()
    x, y = [],[]
    for line in content:
    	data = line.strip().split(',')
    	features = data[1:28]
    	label = data[-1]
    	x.append(features)
    	y.append(label)

    set_name = dataset.split('_')[0]
    np.save(set_name+'_x',np.array(x))
    np.save(set_name+'_y',np.array(y))

read_csv_into_numpy('test_data.csv')