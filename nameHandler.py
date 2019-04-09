import csv

def get_name(id_set, mode, seed, prop='1'):
    '''
        Infers the classical name for csv file (without the extension) according to the parameter (see read me)
    '''
    
    classical_name = id_set + '-'

    # The zero mode is the default mode.
    if mode != 0:
        classical_name += '_' + mode

    classical_name += 's' + str(seed) + ('p' + str(prop) if str(prop) != '1' else '')

    return classical_name

def get_classical_model(csv_paths, arch_type, seed, prop=1, weighted=True):
    '''
        Infers the model of the classical model trained by using the csv_file in the csv_paths, list of strings
        and the architecture type. (see read-me)
        
        csv_path = ['0_1', '0_7', '0_5', '0_7', '0_1', '0_1', '0_5']
        return 0_1_7_5_7_1_1_5&&0_1_0_0_0_0_0 (where the first 0 is for the classical.)
    '''

    classical_name = '0' if arch_type == 'resnet50' else arch_type + '_0'
    id_train_set = []
    modes = []
    only_zero = True
    for csv_path in csv_paths:
        csv_path = csv_path.split('/')[-1]
        csv_path = csv_path.split('-')[0]

        if '&&' in csv_path: #mode = 1
            modes.append(1)
            id_train_set.append(csv_path.split('&&')[0])
            only_zero = False
        else:
            modes.append(0)
            id_train_set.append(csv_path)

    sorted_index = np.argsort(id_train_set)
    id_train_set = id_train_set[sorted_index]
    modes = modes[sorted_index]

    classical_name += '_' + '_'.join(id_train_set) + ('' if only_zero else '&&' + '_'.join(modes))
    classical_name += '-' + ('w' if weighted else '') + 's' + str(seed) + ('p' + str(prop) if prop != 1 else '')

    return classical_name

def get_classical_matrix(csv_train_paths, csv_test_paths, arch_type, seed, prop=1, weighted=True):
    '''
        Infers the confusion matrix name from the csv_file in csv_train_paths and csv_test_paths, list of string
        and the architecture type. (see read-me)
    '''

    classical_name = '0' if arch_type == 'resnet50' else arch_type+'_0'
    id_train_set = []
    modes_train = []
    only_zero_train = True
    
    id_test_set = []
    modes_test = []
    only_zero_test = True

    for csv_path in csv_train_paths:
        csv_path = csv_path.split('/')[-1]
        csv_path = csv_path.split('-')[0]        
        if '&&' in csv_path: #mode = 1
            modes_train.append(1)
            id_train_set.append(csv_path.split('&&')[0])
            only_zero_train = False
        else:
            modes_train.append(0)
            id_train_set.append(csv_path)

    for csv_path in csv_test_paths:
        csv_path = csv_path.split('/')[-1]
        csv_path = csv_path.split('-')[0]        
        if '&&' in csv_path: #mode = 1
            modes_test.append(1)
            id_test_set.append(csv_path.split('&&')[0])
            only_zero_test = False
        else:
            modes_test.append(0)
            id_test_set.append(csv_path)
        
    sorted_index_train = np.argsort(id_train_set)
    sorted_index_test = np.argsort(id_test_set)
    
    id_train_set = id_train_set[sorted_index_train]
    modes_train = modes_train[sorted_index_train]
    id_test_set = id_test_set[sorted_index_test]
    modes_test = modes_test[sorted_index_test]    

    classical_name += '_' + '_'.join(id_train_set) + '&' + '_'.join(id_test_set) + ('' if only_zero else '&&' + '_'.join(modes_train)) + ('' if only_zero else '&&&' + '_'.join(modes_test))
    classical_name += '-' + ('w' if weighted else '') + 's' + str(seed) + ('p' + str(prop) if prop != 1 else '')

    return classical_name


def get_mapping(csv_path):
    '''
        Extracts the mapping between the dataset name et the dataset id.
        Parameters:
            -csv_path: string, path to the csv file that contains the mapping
        Returns:
            - A dictionary where the keys is the name the datasets and the value is the corresponding id.
    '''
    mapping = {}
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            mapping[row['name']] = row['id']

    return mapping
#print(get_classical_name('vgg', ['0','1'], ['5','7','9'], ['0.1','0.6']))
'''
print(get_classical_model(['0_1', '0_7&&1', '0_5', '0_7', '0_1', '0_1', '0_5'], 'resnet50', 7,0.1))
print(get_classical_matrix(['0_1', '0_7&&1', '0_5', '0_7', '0_1', '0_1', '0_5'], ['0_1', '0_7&&1', '0_5', '0_7', '0_1', '0_1', '0_5'], 'resnet50', 7,0.1))
'''