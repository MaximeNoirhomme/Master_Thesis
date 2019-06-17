from dataHandler import DataHandler
import myUtils as mu
import numpy as np

class ImageScraper:
    def __init__(self, labels, nb_images, chromedriver, saved=True, path = 'artistic_instrument', mapping=None):
        # List of labels to scrap
        self.labels = labels
        # Either a list of integer of same size of labels or an integer. If integer, the number
        # of image to scrap is the same for each label.
        if nb_images is list and not len(labels) == len(nb_images):
            raise ValueError('If the field nb_images is a list, then it has to have the same length than the field labels.')
        elif nb_images is list:
            self.nb_images = np.array(nb_images)
        else:
            self.nb_images = np.full(len(labels), nb_images)

        # If set to true, the downloaded image are stored in the folder "path\label"
        self.saved = saved
        self.response = google_images_download.googleimagesdownload()
        self.path = path
        self.mapping = mapping
		
		self.chromedriver = chromedriver

    def download_image(self):
        absolute_img_paths = {}
        for nb_images, label in zip(self.nb_images, self.labels):
            self.response = google_images_download.googleimagesdownload()

            keyword = label + " in painting"#'artistic ' + label
            if self.mapping != None:
                label = self.mapping[label]
            path = self.path + '/' + label

            absolute_img_paths[label] = self.response.download({'keywords':keyword, 'limit':nb_images, 'output_directory':self.path, 'chromedriver':self.chromedriver})[keyword]

        return absolute_img_paths