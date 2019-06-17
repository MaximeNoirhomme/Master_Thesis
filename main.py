import argparse
import ModelReTrainer as mrt
from imageScraper import ImageScraper 
from ModelReTrainer import ModelReTrainer, LAST_LAYER_TRAIN_MODE, FULL_TRAIN_MODE
from keras.applications.resnet50 import ResNet50
from dataHandler import DataHandler
from dataSplitter import DataSplitter
import tensorflow as tf 
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from keras.applications import inception_v3
import os
import subprocess
import myUtils as mu
import numpy as np
from keras.optimizers import SGD
import plot
from keras.applications import resnet50
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import nameHandler as nh
from dataSplitter import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_NAMES = ['la_muse', 'rain_princess', 'the_scream', 'udnie', 'wave', 'shipwreck', 'first_mimo', 'second_mimo', 'third_mimo', 'fourth_mimo', 'first-2']
LABEL = ['2206', '2231', '2451', '2805', '2807', '2861', '2879', '2912', '3062', '3102', '3224', '3265', '3509', '3564', '3731', '3786', '3827', '3859', '3866', '3875', '4318', '4341', '4350', '4455', '4479']
MODEL_SEEDS = [7, 17, 27, 37, 47, 57]
STYLE_ID = set(mu.get_stylized_id())
DATASET_NAMES = mu.get_dataset_names() #['mimo', 'google', 'other', 'style'] + MODEL_NAMES 

SIMPLE_TF = 0 # experience type corresponding to simple transfert learning
DART_TF = 1 # experience type corresponding to DART transfert learning

MATCH_NAME_ID = 'convention_names/dataset_mapping.csv'
DATASET_CSV_PATH = 'dataset_csv_path'
WEIGHTS_MODEL_PATH = 'weights2'
CHECKPOINTS_PATH = 'Checkpoints'
LOGGER_PATH = 'csvLogger'
MATRIX_PATH = 'conf_matrix'
FIGURE_PATH = 'figure'
SIMPLE_TF_MODES = {'mimo':0, 'mimo_la_muse1':1}

def get_arch(name):
    '''
        Load the architecture model corresponding to 'name' and load the pre-trained model on imagenet weights
        Parameters:
            - name: string, name of the architecture (ex: 'resnet')
    '''
    if name == 'resnet50':
        model = ResNet50(weights='imagenet')
    elif name == 'vgg19':
        model = VGG19(weights='imagenet')
    else:
        model = inception_v3.InceptionV3(weights='imagenet')

    return model

def add_generic_argument(parser):
    parser.add_argument('--arch', choices=['resnet50', 'vgg19', 'inceptionv3'], default='resnet50',
                                    help='--arch XXX: Choose the model architecture.'
                                    'XXX is the name of the architecture (ex: resnet).'
                                    'If not set, resnet is set by default.')    

    parser.add_argument('--trainset', nargs="+", choices=DATASET_NAMES, default=['mimo'],
                            help='--testset XXX: Choose the training/validation sets.'
                            'XXX carracterises the testing set used, XXX could be:\n'
                            '\t- mimo: A part of the mimo natural_instru dataset is used for the testing\n'
                            '\t- google: testing set is fetched from google images\n'
                            '\t- other: testing set is\n'
                            '\t- style: all the style transfert dataset from mimo are used (a specific style could be selected by '
                            'write this name, see read me for the list of styled model (ex: la_muse).'
                            'If not set, mimo is set by default.')
    
    parser.add_argument('--testset', nargs="+", choices=['mimo', 'google', 'other', 'style'] + MODEL_NAMES, default=['mimo'],
                            help='--testset XXX: Choose the testing sets.'
                            'XXX carracterises the testing set used, XXX could be:\n'
                            '\t- mimo: A part of the mimo natural_instru dataset is used for the testing\n'
                            '\t- google: testing set is fetched from google images\n'
                            '\t- other: testing set is\n'
                            '\t- style: all the style transfert dataset from mimo are used (a specific style could be selected by '
                            'write this name, see read me for the list of styled model (ex: la_muse).'
                            'If not set, mimo is set by default.')

    parser.add_argument('--mode', choices=['0','1'], nargs='+',
                            help='--mode XXX: indicates either the corresponding training dataset is splitted (train, valid, test) or '
                            'not.\n Mode 0 is the splitted version and 1 is the other one. If not set, the mode of all used dataset is 0.\n'
                            ' If you want to generate the csv split, the mode value is taken into account.')

    parser.add_argument('--mode_test', choices=['0','1'], nargs='+',
                            help='--mode_test XXX: indicates either the corresponding testing dataset is splitted (train, valid, test) or '
                            'not.\n Mode 0 is the splitted version and 1 is the other one. If not set, the mode of all used dataset is 0.')
    
    parser.add_argument('--csv_train_path', nargs='+',
                            help='--csv_train_path XXX: give a list of explicit names for csv file, without the extension, '
                            'that contains the split of train and validation.')

    parser.add_argument('--csv_test_path', nargs='+',
                            help='--csv_train_path XXX: give a list of explicit names for csv file, without the extension, '
                            'that contains the split of testing.')

    parser.add_argument('--prop', nargs='+', default=['0.1'],
                            help='--prop XXX: give the proportion of the training MIMO dataset that has been style transfered to '
                            'train the model.')

    parser.add_argument('--prop_test', nargs='+', default=['0.1'],
                            help='--prop XXX: give the proportion of the training MIMO dataset that has been style transfered to '
                            'train the model.')

    parser.add_argument('--seed', type=int, default=[7])

def get_csv(args, exp_type):
    '''
        Infers from the arguments the paths of csv file that contains all the needed information
        for the training and testing part.
        Parameters:
            - args: a argparse.Namespace object.
            - exp_type: int or None, the experience type. 
    '''

    # First check if some have been explictly gave by the user. 
    csv_train_paths = [] if args.csv_train_path == None else [DATASET_CSV_PATH + '/' + p for p in args.csv_train_path]
    csv_test_paths = [] if args.csv_test_path == None else [DATASET_CSV_PATH + '/' + p for p in args.csv_test_path]
    # Then check if some have been indirectly gave by the user.
    if args.trainset != None:
        name_to_id = nh.get_mapping(MATCH_NAME_ID)
        if exp_type == SIMPLE_TF: # If it is a classical/simple experiment.
            # Get mode list, if not set, it is a list of 0.
            modes_train = np.zeros((len(name_to_id), 1)) if args.mode == None else args.mode
            modes_test = np.zeros((len(name_to_id), 1)) if args.mode_test == None else args.mode_test
            
            for train_set, mode in zip(args.trainset, modes_train):
                id_train = name_to_id[train_set]
                prop_train = args.prop[0] if id_train in STYLE_ID else '1'

                csv_train_paths.append(DATASET_CSV_PATH + '/' + nh.get_name(id_train, mode, args.seed[0], prop_train)) 

            for test_set, mode in zip(args.testset, modes_test):
                id_test = name_to_id[test_set]
                prop_test = args.prop_test[0] if id_test in STYLE_ID else '1'
                csv_test_paths.append(DATASET_CSV_PATH + '/' + nh.get_name(id_test, mode, args.seed[0], prop_test))

    return csv_train_paths, csv_test_paths

def parse_args():
    # Create the parser.
    parser = argparse.ArgumentParser(description='Bench of codes associated to a master thesis in relation to machine learning in art')
    
    subparsers = parser.add_subparsers(dest='modeltype', help='Choose the type of model to train or to use.')
    subparsers.required = True
    
    # Create the parser for training model by using classical architecture as resnet, vgg, etc ...
    nip = subparsers.add_parser('classic', help='stand for model trained on image of instruments (natural, artistic or both according to the dataset')

    nip.add_argument('--otherarch', nargs=1,
                                    help='--otherarch XXX: use a self made architecture, '
                                    'XXX: path where to find the model, if set the \'--arch\' argument is ignored.')

    nip.add_argument('--loadmodel', nargs=1,
                                    help='--loadmodel XXX: Load a pre-trained model, '
                                    'XXX: string, relative path in the folder models \n'
                                    'If not set, the imagenet weight is used instead.')

    nip.add_argument('--loadweights', nargs=1,
                                    help='--loadweights')
    
    nip.add_argument('--nbgoogleimg', type=int, nargs=1,
                        help='--nbgoogleimg XXX: To indicate how many google images to fetch on google.'
                        'XXX is a positive integer indicates the number of images to fetch.'
                        'This field is ignored if the testset is different than \'google\'')

    nip.add_argument('--trainermode', choices=['last', 'all'], default='all',
                                    help='--trainermode XXX: Choose the mode of retraining.'
                                    'XXX is the mode of retraining, XXX could be:\n'
                                    '\t- last: only the last layer is retrained\n'
                                    '\t- all: the entire network is retrained.\n'
                                    'If not set, all is set by default.')

    nip.add_argument('--startepoch', nargs=1, default=[0],
                                    help='--startepoch XXX: indicates that the training start at the epoch XXX, XXX is a positive integer.')

    nip.add_argument('--trainoff', type=bool, nargs='?',
                                    help='--trainoff: disable the training phase')

    nip.add_argument('--savemodeloff', default=False, type=bool, nargs='?',
                                    help='--savemodeloff: disable saving model weights')

    nip.add_argument('--checkpointsoff', default=False, type=bool, nargs='?',
                                help='--checkpointsoff: disable making checkpoints')

    nip.add_argument('--loggeroff', default=False, type=bool, nargs='?',
                                help='--loggeroff: disable logger')

    nip.add_argument('--lr', nargs=1, type=float, default=0.001)

    nip.add_argument('--momentum', type=float, nargs=1, default=0.9)

    nip.add_argument('--decay', nargs=1, type=float, default=0.0)

    nip.add_argument('--batch_size', nargs=1, type=int, default=32)

    nip.add_argument('--matrix_path', nargs=1, type=str)

    nip.add_argument('--output_folder', nargs=1, type=str)

    # Create subparser for visualize what network see
    vp = subparsers.add_parser('visu', help='stand for visualizing model')

    vp.add_argument('--visu_mode', choices=['visu_network'], required=True)
    
    vp.add_argument('--model_names', nargs='+', type=str)

    vp.add_argument('--model_archs', nargs='+', type=str)

    vp.add_argument('--model_weights', nargs='+', type=str)

    vp.add_argument('--folder', nargs=1)
    
    vp.add_argument('--label', nargs=1)

    vp.add_argument('--folder_output', nargs=1)

    vp.add_argument('--not_well_predicted', nargs='?', type=bool, 
                        help='--not_well_predicted XXX: XXX is a bool. If set to True, we only visualize wrong predicted image that is confident enough with respect to alpha, otherwise it visualise '
                        'well predicted image')

    vp.add_argument('--alpha', type=float, help='--alpha XXX: XXX a float number between 0 and 1, it is the confident number. This parameter is ignored if well_predicted == True')

    # Create subparser for plotting results
    pp = subparsers.add_parser('plot', help='stand for ploting results')

    pp.add_argument('--plot_mode', choices=['compute_conf_matrix', 'conf_matrix', 'table_error_label', 'plot_error_label'], required=True,
                    help='--plot_mode XXX: indicate what to plot/display '
                        'XXX: a string, the name of the corresponding option\n.'
                        '- conf_matrix: display the confusion matrix corresponding to the model and the test dataset given by the user\n'
                        '- table_error_label: display a table that indicates for each label, the total error made' 
                        'and the two main wrong predicted label.\n'
                        '- plot_error_label: plot a barplot that indicates for each label, the total error made'
                        'and the two main wrong predicted label.')

    pp.add_argument('--exp_type', choices=['classical_exp', 'dart_exp'], required=True)

    pp.add_argument('--output_folder', nargs=1, default='plot_name')

    pp.add_argument('--trainermode', choices=['last', 'all'], default='all',
                                    help='--trainermode XXX: Choose the mode of retraining.'
                                    'XXX is the mode of retraining, XXX could be:\n'
                                    '\t- last: only the last layer is retrained\n'
                                    '\t- all: the entire network is retrained.\n'
                                    'If not set, all is set by default.')

    pp.add_argument('--weights', type=str)

    pp.add_argument('--matrix_name')

    # Create subparser for splitting data and stores the split information into csv and for style transfered images.
    dhp = subparsers.add_parser('dataHandling', help='stand for handling splitting and style transfered images')
    
    dhp.add_argument('--type_task', choices=['style_transfer', 'splitter', 'add_dataset'], required=True)

    dhp.add_argument('--createstyledataset', nargs=2,
                                    help='--createstyledataset XXX YYY: create a styled dataset, ' #  based on the dataset located at datasetpath
                                    'XXX is the path where to find the original dataset'
                                    'YYY is the path where to create it. Notice that if set, all the train parameter except --datasetpath '
                                    'are ignored.')

    dhp.add_argument('--propsplit', type=float, nargs=1,
                        help='--propsplit XXX: indicates the proportion of the dataset to style-transfer (select randomly according to the seed), '
                        'XXX: number between 0 and 1, the proportion of the dataset. If not set, the proportion is set to 100\% \n'
                        'Notice that this parameter is ignored if --createstyledataset is not set.')

    dhp.add_argument('--model_name', choices=MODEL_NAMES + ['all'], default='la_muse',
                        help='--model_name XXX: indicates which model to use to perform the style-transfer, '
                        'XXX: string of the model. When set to \'all\', all the model are used.')

    dhp.add_argument('--pathsplit', nargs=1)

    dhp.add_argument('--prefixname', nargs=1)

    dhp.add_argument('--csv_name', nargs=1)

    dhp.add_argument('--datasetpath', nargs=1)

    dhp.add_argument('--new_dataset', nargs=1)

    dhp.add_argument('--styled', nargs=1)

    parsers = [nip, vp, pp, dhp]
    [add_generic_argument(p) for p in parsers]

    args = parser.parse_args()
    if args.modeltype == 'classic':
        classic_parser(args)
    elif args.modeltype == 'visu':
        visu_parser(args)
    elif args.modeltype == 'plot':
        plot_parser(args)
    elif args.modeltype == 'dataHandling':
        data_parser(args)

def data_parser(args):
    if args.type_task == 'style_transfer':
        if args.createstyledataset != None:
            path_input = args.createstyledataset[0]
            path_output = args.createstyledataset[1]
            path_models = 'hwalsuklee/models'
            path_hwalsuklee = 'hwalsuklee'
            prop_split = args.propsplit[0] if args.propsplit != None else 1
            if args.model_name == 'all':
                for model_name, seed in zip(MODEL_NAMES, MODEL_SEEDS):
                    folder_output = path_output + '/' + ('' if args.prefixname == None else args.prefixname[0]) + model_name
                    mu.create_folder_if_not_exist(folder_output)
                    mu.create_img_w_rdm_styles(path_input, folder_output, path_models, path_hwalsuklee, prop_split, model_name=model_name, seed=seed)
            else:
                mu.create_img_w_rdm_styles(path_input, path_output, path_models, path_hwalsuklee, prop_split, model_name=args.model_name)
        else:
            raise ValueError('The argument \'createstyledataset\' not found which is needed in order to style transfer a dataset')

    elif args.type_task == 'splitter':
        mode = 0 if args.mode == None else args.mode[0]
  
        # Check if the user already give the name of the csv_file
        if args.csv_name != None:
            csv_name = args.csv_name[0]
        # Otherwise, infer it !
        else:
            name_to_id = nh.get_mapping(MATCH_NAME_ID)
            id_train = name_to_id[args.trainset[0]]
            prop_train = args.prop if id_train in STYLE_ID else 1
            csv_name = nh.get_name(id_train, mode, args.seed[0], prop_train)

        arg = {}
        if mode == 1 or mode == '1':
            args['learn_prop'] = 0
            args['valid_prop'] = 0
            args['test_prop'] = 1

        DataSplitter(args.datasetpath[0], DATASET_CSV_PATH + '/' + csv_name, **arg).split()

    elif args.type_task == 'add_dataset':
        if args.new_dataset== None:
            raise ValueError("In order to add a new dataset, the name of the new dataset has to be mentionned.")
        
        styled = 0 if args.styled == None else args.styled[0]
        mu.add_id(args.new_dataset[0], str(styled))

def plot_parser(args):
    if args.exp_type == 'classical_exp':
        csv_paths = get_csv(args, SIMPLE_TF)
        if args.matrix_name == None:
            matrix_name = nh.get_classical_matrix(*csv_paths, args.arch, args.seed[0]) + '.npy'
        else:
            matrix_name = args.matrix_name

        seed = args.seed[0]
        if args.plot_mode == 'conf_matrix':
            matrix = np.load(MATRIX_PATH + '/' + matrix_name)
            print(matrix)
        elif args.plot_mode == 'compute_conf_matrix':
            # Get architecture model
            model = get_arch(args.arch)
            
            # Retrieve the name of csv that contains the splitting.
            csv_train_paths, csv_test_paths = get_csv(args, SIMPLE_TF)
            
            # Init the data handler.
            data_handler = DataHandler(csv_train_paths, csv_test_paths)
            trainer_mode = mrt.LAST_LAYER_TRAIN_MODE if args.trainermode == 'last' else mrt.FULL_TRAIN_MODE
            
            if args.weights == None:
                model_name = nh.get_classical_model(csv_train_paths, args.arch, seed)
            else:
                model_name = args.weights

            path_weight = WEIGHTS_MODEL_PATH + '/' + model_name
            model_retrainer = ModelReTrainer(model, trainer_mode, data_handler, already_weighted=False, path_weight=path_weight,
                 seed=seed, trainoff=True)

            model_retrainer.confusion_matrix(path_matrix=MATRIX_PATH + '/' + matrix_name)
        else:
            testing_dh = DataHandler(*csv_paths)
            label = np.array(testing_dh.get_labels())
            mapping = mu.get_label_from_csv(set(label), 'lookup_data.csv')
            matrix = np.load(MATRIX_PATH + '/' + matrix_name)
            err = plot.compute_error_per_label(label, matrix, mapping=mapping)
            
            if args.plot_mode == 'table_error_label':
                plot.table_error_per_label(label, *err)
            elif args.plot_mode == 'plot_error_label':
                plot.plot_error_per_label(label, *err, FIGURE_PATH + '/' + args.output_folder[0], mapping=mapping)

def visu_parser(args):
    if args.visu_mode == 'visu_network':
        well_predicted = True if (args.not_well_predicted is None or args.not_well_predicted is False) else False
        
        if args.model_names == None or args.model_archs == None or args.folder == None or args.label == None or args.model_weights == None:
            raise ValueError("'visu_network' mode required the parameters 'model_names', 'model_archs', 'model_weights', 'folder' and 'label'")

        if len(args.model_names) != len(args.model_archs or len(args.model_names) != len(args.model_weights)):
            raise ValueError("'model_names', 'model_weights' and 'model_archs' parameters does not have the same size'")

        modelRetrainers = []
        for i in range(len(args.model_names)):
            model = get_arch(args.model_archs[i])
            modelRetrainers.append(ModelReTrainer(model, 0, DataHandler('a', 'a'), path_weight = args.model_weights[i]))

        folder = args.folder[0]
        label = args.label[0]
        for img in os.listdir(folder + '/' + label):

            confidence_lvl = []
            error_label = []
            prob_true_label = []

            to_visualize = True
            for mrt in modelRetrainers:    
                prob = mrt.predict(folder + '/' + label + '/' + img)[0]

                predicted_label_index = np.argmax(prob)
                
                if not well_predicted: # if visualize image well predicted
                    if predicted_label_index != LABEL.index(label): # don't visualize if the prediction is wrong
                        to_visualize = False
                        break
                else:
                    confidence = mu.compute_normalized_entropy(prob)

                    if predicted_label_index == LABEL.index(label) or confidence > args.alpha:
                        to_visualize = False
                        break
                    else:
                        confidence_lvl.append(confidence)
                        error_label.append(LABEL[predicted_label_index])
                        prob_true_label.append(prob[LABEL.index(label)])

            if True:
                original = load_img(folder + '/' + label + '/' + img, target_size=(224,224))
                imgs = [mpimg.imread(folder + '/' + label + '/' + img)]

                for mrt, arch_name in zip(modelRetrainers, args.model_archs):
                    imgs.append(mrt.visualise_network(folder + '/' + label + '/' + img, save=False, im = original, arch_type=arch_name))

                output_path = None 
                if args.folder_output != None:
                    mu.create_folder_if_not_exist(args.folder_output[0])
                    output_path = args.folder_output[0] + '/' + img
                
                names = args.model_names.copy()

                for i, (conf_lvl, err_label, prob) in enumerate(zip(confidence_lvl, error_label, prob_true_label)):
                    names[i] += ':\n a = ' + str(conf_lvl) + ', b = ' + str(err_label) + ', c = ' + str(prob)
                
                imgs[0] = original
                plot.plot_cmp_img('Comparaison of what networks visualize', imgs, ['original'] + names, output_path=output_path)

def classic_parser(args):     
    model = get_arch(args.arch)

    test_only = False if (args.trainoff is None or args.trainoff is False) else True #
    checkpoint_enable = True if args.checkpointsoff is None or args.checkpointsoff is False else False 
    logger_enable = True if args.loggeroff is None or args.loggeroff is False else False
    save_model_enable = True if args.savemodeloff is None or args.savemodeloff is False else False
    
    # Initialise the dataHandler for the training/validation phase (and possibly testing phase)
    args_dataset = {}
    seed = args.seed[0] 
    args_dataset['seed'] = seed    
    args_dataset['batch_size'] = args.batch_size

    # Retrieve the name of csv that contains the splitting.
    csv_train_paths, csv_test_paths = get_csv(args, SIMPLE_TF)
    # Init the data handler.
    data_handler = DataHandler(csv_train_paths, csv_test_paths, **args_dataset)

    # Retrain part.
    trainer_mode = mrt.LAST_LAYER_TRAIN_MODE if args.trainermode == 'last' else mrt.FULL_TRAIN_MODE
    model_name = nh.get_classical_model(csv_train_paths, args.arch, seed)

    # Check if the model has been already trained
    if os.path.isfile(WEIGHTS_MODEL_PATH + '/' + model_name):    
        already_weighted = False
        path_weight = WEIGHTS_MODEL_PATH + '/' + model_name
    elif test_only:
        raise ValueError('The specified model(' + model_name + ') has not been trained yet and the train is off, train the model'
        ' before testing it.')
    else:
        already_weighted = True
        path_weight = None

    path_save = None if not checkpoint_enable else CHECKPOINTS_PATH + '/' + model_name
    path_logger = LOGGER_PATH + '/' + model_name + '.csv' if logger_enable else None
    
    model_retrainer = ModelReTrainer(model, trainer_mode, data_handler, already_weighted=already_weighted, path_weight=path_weight, path_save=path_save,
                 seed=seed, optimizer=SGD(lr=args.lr, momentum=args.momentum, decay=args.decay), path_logger=path_logger)

    if not test_only:
        history = model_retrainer.train(int(args.startepoch[0]))

    if save_model_enable:
        model_retrainer.save(WEIGHTS_MODEL_PATH + '/' + model_name)
    
    # testing phase (computation of confusion_matrix)
    matrix_name = nh.get_classical_matrix(csv_train_paths, csv_test_paths, args.arch, seed) + '.npy'
    conf_matrix = model_retrainer.confusion_matrix(path_matrix=MATRIX_PATH + '/' + matrix_name)
    err = plot.compute_error_per_label(LABEL, conf_matrix)
if __name__ == "__main__":
    # Parse input
    parse_args()