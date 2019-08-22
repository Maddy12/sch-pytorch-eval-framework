# Datasets
There are a variety of datasets that will have functionality with this framework, each in their own framework and structure. 

## KITTI
### RawData
Step 1. Visit http://www.cvlibs.net/download.php?file=raw_data_downloader.zip and provide an email address to receive the download link.
Step 2. Once you receive the download link in your email and download its contents into the 'data/' directory of this repository.
Step 3. Unzip the file and run in a terminal `sudo ./raw_data_downloader.sh`, thye e download will take a few minutes.
Step 4.

## CityScapes
This dataset is available via `torchvision` and is semantic overview of urban street scenes. 
To get an overview from the source: https://www.cityscapes-dataset.com/dataset-overview/

Step 1. Register or log in to https://www.cityscapes-dataset.com and then proceed to https://www.cityscapes-dataset.com/download
Step 2. Download the following: 
  * https://www.cityscapes-dataset.com/file-handling/?packageID=1
  * https://www.cityscapes-dataset.com/file-handling/?packageID=2
  * https://www.cityscapes-dataset.com/file-handling/?packageID=3
  * https://www.cityscapes-dataset.com/file-handling/?packageID=4
Step 3. Import the 'datasets' module from 'utils' in your script.

