import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
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
        acc = row[i] / nb_sample
        
        # Now search which label the model predict the most instead of the diagonal element.
        sorted_index = np.argsort(row)
        sorted_index = sorted_index[np.where(sorted_index != i)]
        sorted_index = sorted_index[-1:-(nb_label_err + 1):-1]

        for l, k in enumerate(sorted_index):
            error_values[i][l] = row[k] / nb_sample
            error_labels[i][l] = labels[k][1:-1]

        total_error_value.append(1 - acc)

    print('error = ', np.mean(np.array(total_error_value)))

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

