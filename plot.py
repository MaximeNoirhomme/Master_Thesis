import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def compute_error_per_label(labels, confusion_matrix, nb_label_err=2, mapping=None):
    error_labels = np.empty((len(confusion_matrix), nb_label_err), dtype=int)    
    error_values = np.empty((len(confusion_matrix), nb_label_err))
    total_error_value = []

    nb_samples = []
    labels = labels.copy()

    for i, row in enumerate(confusion_matrix):
        # Compute and store the number of samples, of the label corresponding to that row, used to test acc.
        nb_sample = sum(row)
        if nb_sample == 0:
            continue
        nb_samples.append(nb_sample)
        # acc = diagonal element / all the element
        print(nb_sample)
        print(row[i])
        
        acc = row[i] / nb_sample
        
        # Now search which label the model predict the most instead of the diagonal element.
        sorted_index = np.argsort(row)
        sorted_index = sorted_index[np.where(sorted_index != i)]
        sorted_index = sorted_index[-1:-(nb_label_err + 1):-1]

        for l, k in enumerate(sorted_index):
            error_values[i][l] = row[k] / nb_sample
            error_labels[i][l] = labels[k][1:-1]

        total_error_value.append(1 - acc)

    '''np.save('figure/plot_acc/numpy/error_values_'+save_name+'.npy', error_values)
    np.save('figure/plot_acc/numpy/error_labels_'+save_name+'.npy', error_labels)
    np.save('figure/plot_acc/numpy/total_error_'+save_name+'.npy',total_error_value)'''
    print(np.mean(np.array(total_error_value)))

    return error_values, error_labels, total_error_value

def plot_error_per_label(labels, error_values, error_labels, total_error_value, output_folder, width=0.3, height_text = 0.01, nb_label_per_plot=5, mapping=None):
    nb_labels = len(labels)
    nb_plots = nb_labels //  nb_label_per_plot + 1

    error_values = error_values.copy()
    error_labels = error_labels.copy()

    for j in range(nb_plots):
        nb_bars = min([(j + 1) * nb_label_per_plot, nb_labels]) - j * nb_label_per_plot # The number of labels for this plot
        
        if nb_bars == 0:
            break

        ind = np.arange(nb_bars)
        
        plt.bar(ind, error_values[:, 0][:nb_bars], width)
        plt.bar(ind + width, error_values[:, 1][:nb_bars], width)
        plt.bar(ind + 2*width, total_error_value[:nb_bars], width)
        plt.xticks(ind + width, labels=labels[:nb_bars] if mapping == None else [mapping[l] for l in labels[:nb_bars]], size=6)
        plt.ylim((0,1))
        for i in range(nb_label_per_plot):
            f_label = str(error_labels[:, 0][i]) if mapping == None else mapping[str(error_labels[:, 0][i])]
            s_label = str(error_labels[:, 1][i]) if mapping == None else mapping[str(error_labels[:, 1][i])]
            plt.text(x = ind[i], y = error_labels[:, 0][i] + len(f_label) * 0.014, horizontalalignment='center', rotation=90, s=f_label, size=6)
            plt.text(x = ind[i] + width , y = error_labels[:, 1][i] + len(s_label) * 0.014, horizontalalignment='center', rotation=90, s=s_label, size=6)
            plt.text(x = ind[i] + 2*width , y = total_error_value[i] + 0.035, horizontalalignment='center', rotation=90, s='total', size=6)
            plt.text(x=ind[i] + width, y = 0, horizontalalignment='center', s=nb_samples[i])

        error_values = error_values[nb_label_per_plot:] 
        error_labels = error_labels[nb_label_per_plot:]
        total_error_value = total_error_value[nb_label_per_plot:]

        labels = labels[nb_label_per_plot:]
        nb_samples = nb_samples[nb_label_per_plot:]

        plt.savefig(output_folder + '/' +str(j))
        plt.close()
        plt.show()

def plot_cmp_img(global_title, imgs, titles, output_path=None):
    if len(imgs) != len(titles):
        raise ValueError("The number of images does not match the number of" + 
                         " title. Find (" + str(len(imgs)) + ", " +
                         str(len(titles)) + ')')

    fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
    axs = [fig.add_subplot(1, len(imgs), i+1) for i in range(len(imgs))]

    #fig, axs = plt.subplots(1, len(imgs), constrained_layout=True)
    for i in range(len(imgs)):
        if i != 0:
            img = np.sum(np.abs(imgs[i]), axis=2)
            vmax = np.percentile(img, 99)
            vmin = np.min(img)
            axs[i].imshow(img, vmin=vmin, vmax=vmax)

        else:
            img = imgs[0]
            axs[i].imshow(img)
        #axs[i].imshow(imgs[i])
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    fig.suptitle(global_title, fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    if output_path == None:    
        plt.show()
    else:
        plt.savefig(output_path)

def plot_curves_from_csv(csv_path, attribute_names, curve_names):
    nb_subplots = len(attribute_names)
    
    if nb_subplots == 1:
        fig = plt.figure()
        subplots = [fig.add_subplot()] 
    elif nb_subplots == 2:
        fig = plt.figure()
        subplots = [fig.add_subplot(2,1,1), fig.add_subplot(2,1,2)]
    elif nb_subplots == 3:
        fig, subplots = plt.subplots(nrows=2, ncols=2)
        subplots = [ax for axes in subplots for ax in axes] #tuple to flatten list
        # only need 3 plots
        subplots[-1].set_visible(False)
        subplots = subplots[0:3] 
    elif nb_subplots == 4:
        fig, subplots = plt.subplots(nrows=2, ncols=2)
        subplots = [ax for axes in subplots for ax in axes] #tuple to flatten list

    [(subplot.set_xlabel('epoch'), subplot.set_ylabel('loss')) for subplot in subplots]

    for i in range(nb_subplots):
        curves = [[] for j in range(len(attribute_names[i]))]
        nb_epochs = []
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nb_epochs.append(row['epoch'])
                for j, attribute_name in enumerate(attribute_names[i]):
                    curves[j].append(float(row[attribute_name]))

        for j in range(0, len(curves)):
            subplots[i].plot(nb_epochs, curves[j], label=curve_names[i][j])

        subplots[i].legend()

    fig.subplots_adjust(hspace=0.8)        
    plt.tight_layout()
    plt.show()

def table_error_per_label(labels, error_values, error_labels, total_error):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')    
    
    rows = labels
    columns = ['total_error', 'first_main_error', 'second_main_error']

    first_error = error_values[:, 0]
    first_label = error_labels[:, 0]

    second_error = error_values[:, 1]
    second_label = error_labels[:, 1]

    cell_text = []
    for i in range(len(rows)):
        cell_text.append([str(total_error[i]), str(first_label[i]) + ' : ' + str(first_error[i]), str(second_label[i]) + ' : ' + str(second_error[i])])
    
    the_table = plt.table(cellText=cell_text,
                    rowLabels=rows,
                    colLabels=columns,
                    colWidths=[0.2 for x in columns],
                    loc='center',
                    cellLoc='center',
                    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1,1)
    fig.tight_layout()
    plt.show()


# val_dom_class_acc,val_dom_class_c_w_acc,val_dom_class_loss,val_lab_class_acc,val_lab_class_c_w_acc,val_lab_class_loss,val_loss
#plot_curves_from_csv('csvLogger/Dart_google_11.csv', [['val_dom_class_loss'],['val_lab_class_loss', 'val_loss']], [['val_dom_class_loss'],['val_lab_class_loss', 'val_loss']])
#


'''data = [['87.83', '87.01', '86.24', '', ''], ['17,4', '21.4', '23.8', '','24'], ['5.1','68.76','74.04','40','76.7'], ['7.68','43.21','75.76', '13.32', '79.43'], ['7.7','57.68','81.26', '36.02', '82.96'], ['12.18','47.84','78.09','27.89', '81,36'], ['11.28','39.06','79.03', '21.07', '81.67'], ['15.79','52.87','80.15', '36.21', '82.38']]
columns = ['\'MIMO\' model', '\'MIMO + la_muse\' model', '\'MIMO + the 6 styles\' model', '\'Weighted MIMO\' model', '\'Weighted MIMO + the 6 styles\' model']
rows = ['acc on MIMO', 'acc on google', 'acc on la_muse', 'acc on rain princess', 'acc on the scream', 'acc on udnie', 'acc on wave', 'acc on shipwrek']

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')


index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))
n_rows = len(rows)
cell_text = []
for row in range(n_rows):
    cell_text.append(data[row])

print(cell_text)
print(len(columns))
print(len(rows))
print(len(data))
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      colWidths=[0.2 for x in columns],
                      loc='center',
                      cellLoc='center',
                      )
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1,1)
fig.tight_layout()
plt.show()'''
    
