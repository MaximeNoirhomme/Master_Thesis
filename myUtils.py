import os, errno
import sys
import subprocess
import signal
import psutil
from PIL import Image
import random
from keras.preprocessing.image import load_img
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil 
import math 
from matplotlib import offsetbox
import seaborn as sns
import pickle

sns.set()

def create_folder_if_not_exist(folder_name):
    '''
        create the folder folder_name if don't exist (this subpart of the code comes from:
        https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python)
        Parameters:
            - folder_name: name of the folder to create if don't already exists.
    '''
    try:
        os.makedirs(folder_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def copyWithSubprocess(source, dest):
    '''
        Copy a file from source to dest, this code come from:
        https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
        and https://stackoverflow.com/questions/22078621/python-how-to-copy-files-fast.
        Parameters:
            - source: path where the file is
            - dest: path where to copy the file.
    '''        

    if sys.platform.startswith("darwin"): cmd=['cp', source, dest]
    elif sys.platform.startswith("win"): cmd=['xcopy', source, dest, '/K/O/X']

    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_pid = pro.pid
    try:
        pro.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


def check_dataset(path):
    # from https://opensource.com/article/17/2/python-tricks-artists
    for label in os.listdir(path):
        for img in os.listdir(path + '/' + label):
            try:
                img_ = Image.open(path + '/' + label + '/' + img) # open the image file
                img_.verify() # verify that it is, in fact an image
                img_ = Image.open(path + '/' + label + '/' + img) # open the image file
                img_.load()
            except (IOError, SyntaxError) as e:
                print('Bad file:', path + '/' + label + '/' + img) # print out the names of corrupt files
            except (IOError) as e:
                print('truncatate file:', path + '/' + label + '/' + img)


def create_img_w_rdm_styles(path_input, path_output, path_models, path_hwalsuklee, prop_split, add_style_name = False,
        model_names = ['la_muse', 'rain_princess', 'the_scream', 'udnie', 'wave', 'shipwreck'], model_name = 'la_muse', seed=7):
    '''
        Use the code of https://github.com/hwalsuklee/tensorflow-fast-style-transfer in order to perfom a style-transfer
        on an image. The style used for the style transfer is chosen randomly among those specified in model_names.
        Parameters:
            - path_input: path where to find the original image.
            - path_output: path where to save the stylised image.
            - path_models: path where the models have been stored.
            - path_hwalsuklee: path where the hwalsuklee github folder is.
            - add_style_name: boolean, if true then '-style_use' will be added, 
            at the end of the output name just before the extension .
            - model_names: List of strings where each element is a name of models.
        
        Notice that if a model thats is not in the folder located to path_models has its name in model_names, it won't work !
        If a model is present n times in the model_names, the probability that it is taken is n/l where l is the length of
        model_names.
    '''

    model_str = '/' + model_name + '.ckpt'

    cmd = 'python ' + path_hwalsuklee + '/run_test.py --content_folder ' + path_input + ' --style_model ' + path_models + model_str + ' --folder_output ' + path_output + ' --style ' + model_name + ' --random ' + str(prop_split) + ' --seed ' + str(seed) 
    subprocess.call(cmd.split())

def value2key(my_map, value):
    '''
        Finds the key corresponding to the value 'value' in map my_map. 
        For example: if my_map = {'a':0, 'b':1}, then value2key(my_map, 1) returns 'b'.
        Parameters:
            - my_map: A dictionnary
            - value: a value contains in my_map
        Returns:
            - The key corresponding to the 'value' in map my_map.
    '''

    for key in my_map:
        if my_map[key] == value:
            return key

    return None

def get_label_from_csv(prev_labels, csv_path):
    mapping = {}
    nb_labels = len(prev_labels)
    j = 0
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if j >= nb_labels:
                break
            mimo_codes = row['MIMO-Code'][1:-1].split(', ')
            instrument_names = None
            for i, mimo_code in enumerate(mimo_codes):
                if mimo_code in prev_labels and not mimo_code in mapping:
                    if instrument_names == None:
                        instrument_names = row['Instrument-Name'].split(' ')
                    mapping[mimo_code] = instrument_names[i].replace('u\'', '\'').replace(',','').replace('[','').replace('_',' ').replace('\'', '').replace(']','')
                    j += 1
    
    return mapping

def reverse_mapping(mapping):
    return {v:k for k, v in mapping.items()}

def show_image(image, grayscale = False, ax=None, title='', output='visu'):
    '''
        This function does no belong to me, see 
        https://github.com/paintception/Deep-Transfer-Learning-for-Art-Classification-Problems/blob/master/saliency_maps_activations/utils.py
    '''
    if ax is None:
        plt.figure()
    plt.axis('off')
    
    if len(image.shape) == 2 or grayscale == False:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imsave(output, image, vmin=vmin, vmax=vmax)

    else:
        image = image + 127.5
        image = image.astype('uint8')
        
        plt.imsave(output, image)

def crop_images(folder_path, prop_left = None, prop_right = None, prop_up = None, prop_down = None):
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        img = Image.open(folder_path + '/' + img_name)
        width, height = img.size
        
        x = 0 if prop_left == None else prop_left * width
        y = 0 if prop_up == None else prop_up * height
        w = width if prop_right == None else width * (1 - prop_right)
        h = height if prop_down == None else height * (1 - prop_down)

        box = x, y, w, h
        path, extension = os.path.splitext(img_name)
        img.crop(box).save(folder_path + '/' + img_name, 'JPEG')
        #img.crop(box).save(folder_path + '/' + path + '-' + extension, 'JPEG')


def if_scalar_convert_to_list(elm, l, name='elm'):
    '''
        Converts elm into a list of lenght 'l' if it is a scalar.
        Parameters:
            - elm: can be everything.
            - l: integer, size of resulting non-scalar element.
            - name: string, the name of the element (by default 'elm') that is used to print more interpretable error.
        
        Returns: 
            - A non-scalar element of size l.

        Exception:
            - Value error: if elm is not a scalar and it's size is different than l. 
    '''

    if np.isscalar(elm):
        elm = [elm] * l
    elif len(elm) != l:
        raise ValueError(name + ' is not a scalar and do not have the same size than l, indeed ' + str(len(elm)) + ' instead of ' + str(l))
    return elm

def mergefolders(root_src_dir, root_dst_dir):
    '''
        https://lukelogbook.tech/2018/01/25/merging-two-folders-in-python/
    '''
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)

def get_size_csv(csv_path):
    with open(csv_path) as csv_file:
        size = sum(1 for row in csv_file)
    
    return size

def get_stylized_id():
    id_style = []
    with open('convention_names/dataset_mapping.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['stylized'] == '1':
                id_style.append(row['id'])

    return id_style

def get_dataset_names():
    dataset_names = []
    with open('convention_names/dataset_mapping.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset_names.append(row['name'])

    return dataset_names

def add_id(name, styled):
    with open('convention_names/dataset_mapping.csv', 'r+') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['name'] == name:
                raise ValueError("A dataset with the name: '"+name+"' already exists.")

            id_dataset = int(row['id'])

        csvfile.write('\n'+name+','+str(id_dataset + 1)+','+styled)

def compute_normalized_entropy(prob):
    norm_const = math.log2(len(prob))

    value = sum(prob * np.log2(prob))

    return - value / norm_const

def iou_computation(bb1, bb2):
    x_dif = abs(bb1[0] - bb2[0])
    y_dif = abs(bb1[1] - bb2[1])

    w_inter = max([(bb1[2] + bb2[2]) / 2 - x_dif, 0])
    h_inter = max([(bb1[3] + bb2[3]) / 2 - y_dif, 0])
    
    area_inter = w_inter * h_inter

    area_union = bb1[2] * bb1[3] + bb2[2] * bb2[3] - area_inter

    iou = area_inter / area_union

    return iou

def bb_13_to_2(path_input, path_output):
    '''
        Go to yolo labels with 13 labels to 2 (label 7 and other)
    '''
    import shutil
    files = set(os.listdir(path_input))
    first = True
    np_paths = []
    img_paths = []
    ze = 0
    un = 0  
    for f in files:
        name, ext = os.path.splitext(path_input + '/' + f)

        if ext == '.png':
            h = 1
            shutil.copyfile(name+ext, path_output + '/' + f)
        else:
            array = np.load(name + '.npy')
            prob = np.reshape(array[11*11*2*4+11*11*2:], (11,11,13))
            new_prob = np.zeros((11,11,2))
            for i in range(11):
                for j in range(11):
                    if np.sum(prob[i][j]) == 0:
                        continue
                    else:
                        prob_arg = np.argmax(prob[i][j], axis=-1)
                        
                        if prob_arg == 7:
                            ze += 1
                            new_prob[i][j][0] = 1
                        else:
                            un +=1
                            new_prob[i][j][1] = 1

            new_prob = np.reshape(new_prob, (11*11*2))
            new_array = np.concatenate([array[:11*11*2*4+11*11*2], new_prob])
            np.save(path_output + '/' + f,new_array)

    data_to_write = {
        'img_path': img_paths,
        'label': np_paths
    }

    df = pd.DataFrame(data=data_to_write)
    df.to_csv('dataset_csv_path/yolo4.csv', mode='a')

def plot_embedding(X, nb_images, name_imgs, title=None, img_dic=None, dic_name=None):
    '''
        This function does not belong to me: https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
    '''
    plot_wout_dic = img_dic == None
    if plot_wout_dic:
        img_dic = {}
        new_x = None 
    else:
        new_x = []

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(nb_images):
            my_img = load_img(name_imgs[i], target_size=(24,24))

            if plot_wout_dic:
                img_dic[name_imgs[i]] = X[i]

                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(my_img),
                    X[i])
            
            else:
                x_dic = img_dic[name_imgs[i]]
                new_x.append(x_dic)
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(my_img),
                    x_dic)

            ax.add_artist(imagebox)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
    
    if dic_name != None:
        pickle_out = open(dic_name, 'wb')
        pickle.dump(img_dic, pickle_out)
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    return new_x
