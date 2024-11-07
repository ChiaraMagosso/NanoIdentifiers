# NanoIdentifiers

This repository is part of the work "Artificial fingerprints engraved through block-copolymers as nanoscale physical unclonable functions for authentication and identification" by 
Irdi Murataj, Chiara Magosso, Stefano Carignano, Matteo Fretto, Federico Ferrarese Lupi, and Gianluca Milano submitted for publication (https://doi.org/10.21203/rs.3.rs-4170364/v1).

The repository contains 2 files and 2 folders:
- The file named 'QR-Code.py' allows you to generate QR-codes from a binarized SEM image and the relative defect_coordinates file. The file can be run locally.
- The file named 'Matching.py' allows you to match a couple of binarized SEM images. The file can be run locally.
- The folder named 'QR-Code_data' are test data and analysis result for the QR-Cede script.
- The folder named 'Matching_data' are test data and analysis result for the Matching script.

Before running the python files locally make sure you have installed the list of libraries with version reported in the requirements.txt file.
To ran the files correctly on your data change the path according to where the files are located on your pc. The scripts are written for running on the relative path of the data provided in the example.

The script 'sem_image_automated_preprocessing_for_ADAblok.py' to correctly analyze and binarize SEM images is provided at the following link https://github.com/ChiaraMagosso/BlockMetrology. We highly recommend to follow the readme instruction of the BlockMetrology repository for installation and uncomment the line of code where indicated (commented: "# activate only if you are doing PUF"). If the code is used for matching also uncomment the lines of code where indicated (commented: "# activate only if you are doing PUF matching").