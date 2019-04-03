# Master_Thesis: Annotation artistic images with deep learning
## Under the supervision of Prof. Pierre Geurts and Matthia Sabatelli, ULiège 2019
### Objectives:
The objectives of this project is to automatically find bounding boxes of musical instruments and classify them in artistic images. To do so, we used a dataset of artistic instruments from cytomine and used transfer learning from imagenet and MIMO (Musical Instrument Museum Online). Furthermore, we styled transfer image from MIMO in order to make data augmentation. We used "classical" transfer learning and domain adversial method. In more of this, this project aimed to show which parts of the images the model is focused on in order to make prediction. This could help the experts of this field to have intuition of what these networks do (instead of black boxes).
### How to use ?
First, you have to specified what you want to perform, train a model using classical transfer learning (1), using domain adversial (2), show results like confusion matrix (3), visualizing networks (4) or handling dataset (5) ?
To specified it, do:
          python main.py name_task
where name_task is either:
  - classic (1)
  - dan (2)#TODO: make it accesible from main.py
  - plot (3)
  - visu (4)
  - dataHandling (5)
### classic
### dan
TODO: since it is not accessible from main.py for now
### plot
### dataHandling
