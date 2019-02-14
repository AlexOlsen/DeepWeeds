# DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning

This repository makes available the source code and public dataset for the work, "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning", published with open access by Scientific Reports: https://www.nature.com/articles/s41598-018-38343-3. The DeepWeeds dataset consists of 17,509 images capturing eight different weed species native to Australia in situ with neighbouring flora. In our work, the dataset was classified to an average accuracy of 95.7% with the ResNet50 deep convolutional neural network.

The source code, images and annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The contents of this repository are released under an [Apache 2](LICENSE) license.

## Download the dataset images and our trained models

* [images.zip](https://nextcloud.qriscloud.org.au/index.php/s/a3KxPawpqkiorST/download) (892 MB)
* [models.zip](https://nextcloud.qriscloud.org.au/index.php/s/Y7EhlkVMYCqxdg2/download) (561 MB)

Due to the size of the images and models they are hosted outside of the Github repository. The images and models must be downloaded into directories named "images" and "models", respectively, at the root of the repository. If you execute the python script (deepweeds.py), as instructed below, this step will be performed for you automatically.

## Weeds and locations
The selected weed species are local to pastoral grasslands across the state of Queensland. They include: "Chinee apple", "Snake weed", "Lantana", "Prickly acacia", "Siam weed", "Parthenium", "Rubber vine" and "Parkinsonia". The images were collected from weed infestations at the following sites across Queensland: "Black River", "Charters Towers", "Cluden", "Douglas", "Hervey Range", "Kelso", "McKinlay" and "Paluma". The table and figure below break down the dataset by weed, location and geographical distribution.

**Table 1.** The distribution of *DeepWeeds* images by weed species (row) and location (column).
![alt text](https://i.imgur.com/2e0ow8l.png "Distribution of DeepWeeds images by species and location.")

![alt text](https://i.imgur.com/scmJcS3.jpg "Geographical distribution of DeepWeeds images.")
**Figure 2.** The geographical distribution of *DeepWeeds* images across northern Australia  (Data: Google, SIO, NOAA, U.S. Navy, NGA, GEBCO; Image © 2018 Landsat / Copernicus; Image © 2018 DigitalGlobe; Image © 2018 CNES / Airbus).

## Data organization

Images are assigned unique filenames that include the date/time the image was photographed and an ID number for the instrument which produced the image. The format is like so: ```YYYYMMDD-HHMMSS-ID```, where the ID is simply an integer from 0 to 3. The unique filenames are strings of 17 characters, such as 20170320-093423-1.

## labels

The labels.csv file assigns species labels to each image. It is a comma separated text file in the format:
```
Filename,Label,Species
...
20170207-154924-0,jpg,7,Snake weed
20170610-123859-1.jpg,1,Lantana
20180119-105722-1.jpg,8,Negative
...
```

*Note: The specific label subsets of training (60%), validation (20%) and testing (20%) for the five-fold cross validation used in the paper are also provided here as CSV files in the same format as "labels.csv".*

## models

We provide the most successful ResNet50 and InceptionV3 models saved in Keras' hdf5 model format. The ResNet50 model, which provided the best results, has also been converted to UFF format in order to construct a TensorRT inference engine.
```
resnet.hdf5
inception.hdf5
resnet.uff
```

## deepweeds.py

This python script trains and evaluates Keras' base implementation of ResNet50 and InceptionV3 on the DeepWeeds dataset, pre-trained with ImageNet weights. The performance of the networks are cross validated for 5 folds. The final classification accuracy is taken to be the average across the five folds. Similarly, the final confusion matrix from the associated paper aggregates across the five independent folds. The script also provides the ability to measure the inference speeds within the TensorFlow environment.

The script can be executed to carry out these computations using the following commands.

* To train and evaluate the ResNet50 model with five-fold cross validation, use `python3 deepweeds.py cross_validate --model resnet`.
* To train and evaluate the InceptionV3 model with five-fold cross validation, use `python3 deepweeds.py cross_validate --model inception`.
* To measure inference times for the ResNet50 model, use `python3 deepweeds.py inference --model models/resnet.hdf5`.
* To measure inference times for the InceptionV3 model, use `python3 deepweeds.py inference --model models/inception.hdf5`.

## Dependencies

The required Python packages to execute deepweeds.py are listed in requirements.txt.

## tensorrt

This folder includes C++ source code for creating and executing a ResNet50 TensorRT inference engine on an NVIDIA Jetson TX2 platform. To build and run on your Jetson TX2, execute the following commands:
```
cd tensorrt/src
make -j4
cd ../bin
./resnet_inference
```

## Citations

If you use the DeepWeeds dataset in your work, please cite it as:

IEEE style citation: “A. Olsen, D. A. Konovalov, B. Philippa, P. Ridd, J. C. Wood, J. Johns, W. Banks, B. Girgenti, O. Kenny, J. Whinney, B. Calvert, M. Rahimi Azghadi, and R. D. White, “DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning,” *Scientific Reports*, vol. 9, no. 2058, **2** 2019. [Online]. Available: https://doi.org/10.1038/s41598-018-38343-3 ”

## BibTeX
```
@article{DeepWeeds2019,
  author = {Alex Olsen and
    Dmitry A. Konovalov and
    Bronson Philippa and
    Peter Ridd and
    Jake C. Wood and
    Jamie Johns and
    Wesley Banks and
    Benjamin Girgenti and
    Owen Kenny and 
    James Whinney and
    Brendan Calvert and
    Mostafa {Rahimi Azghadi} and
    Ronald D. White},
  title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
  journal = {Scientific Reports},
  year = 2019,
  number = 2058,
  month = 2,
  volume = 9,
  issue = 1,
  day = 14,
  url = "https://doi.org/10.1038/s41598-018-38343-3",
  doi = "10.1038/s41598-018-38343-3"
}

```
