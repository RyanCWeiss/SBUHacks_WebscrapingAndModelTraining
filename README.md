# SBUHacks_WebscrapingAndModelTraining

SBUHacks '21: create labeled training data via scraping google images with selenium + simple classifications with tensorflow + retraining pre-trained image classifier models

##Before getting started gather the following:
Python 3.8+
Tensorflow 2.5.0+
Tensorflow-hub `pip install --update tensorflow-hub`
Selenium: `pip install selenium`
google chrome
chrome webdriver: https://chromedriver.chromium.org/downloads
(must match your version of chrome, put in a location you will remember)

##Goals: 
###functionalities:
Utilize images retireved from google search as data for training an image classifier. This will be implemented via selenium using a chrome webdriver.
Create custom labeled data set using tensorflow builtin features and some workarounds
Utilize pretrained models to help reduce the training time.

###GUI:
allows the user to input new/unique images for the model to classify and display the distributions of certainty in the classification
allows user to retrain model with different classes creating new datasets to train the model from.

GUI Plan:
  file picker:
    pick images to classify
    -> onImageSelection(Str path): store image path
  image field:
    show image
  text input:
    add classes (querying data)
    -> onClick() store classes to list
    -> seleniumGoogleSearch(Str[] queries) on each 
  Display output data
    -> onClassifyImage(Str newImagePath): classificationValues -> displayVals(classificationValues):null
  *graph(ish thing/ distribution)
    show certainty of classification
     -> onClassifyImage(Str newImagePath): classificationValues -> displayValsAsChart(classificationValues):null

