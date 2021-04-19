# Preprocess FinalData_{ppzee|ppttbar}.hdf5, split into train/eval and separate test
test_size = 160000  # This many events will be used to create test set taken from the end of the data (i.e. the last 160,000 events)
import h5py

from utils import get_dataset
dataset_names = ('FinalData_ppzee', 'FinalData_ppttbar')
for dataset_name in dataset_names:
    print('processing', dataset_name)
    dataset = get_dataset(dataset_name)
    print(dataset['z_data'].shape, dataset['x_data'].shape)
    my_dataset_name = dataset_name.split('FinalData_')[-1]  # Just ppzee or ppttbar

    h5f = h5py.File('data/%s.hdf5' %my_dataset_name, 'w')   # Used for training/evaluation
    h5f.create_dataset('FDL', data=dataset['z_data'][:-test_size])
    h5f.create_dataset('ROL', data=dataset['x_data'][:-test_size])
    h5f.close()

    h5f = h5py.File('data/%s_test.hdf5' %my_dataset_name, 'w')  # Separate (unseen) test set for reporting results
    h5f.create_dataset('FDL', data=dataset['z_data'][-test_size:])
    h5f.create_dataset('ROL', data=dataset['x_data'][-test_size:])
    h5f.close()

