# SEAM PUCKERING CLASSIFICATION

## IDEA
This project is created to serve the purpose of classifying the puckering level based on ISO standard as below:

![Seam puckering levels](./sample_data/seam_puckering_level.jpg)

In this project, several models are used to trained and tested including:
* EfficientNet_B0
* RESNET50
* MobileNetV2
* VGG19
* VGG16

## STRUCTURE
* Folder **[ai](https://github.com/ai-4-ia/seam_puckering_classification/tree/main/ai)**: Contain code of notebook to train and test the models
* Folder **[app](https://github.com/ai-4-ia/seam_puckering_classification/tree/main/app)**: Contains the code of building the application utilizing a model for classifying seam puckering job
* Folder **[sample_data](https://github.com/ai-4-ia/seam_puckering_classification/tree/main/sample_data)**: Containing some sample images for each seam puckering level for testing

## AN EXAMPLE OF SOME LABELED SAMPLE IMAGES
![Seam pucker sample images list](./sample_data/seam_image_list.png)

## REQUIREMENT FOR RUNNING APP
Please refer to file **_requirements.txt_** for configuring suitable library to deploy or run the app locally
