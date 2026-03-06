# FAZ_Segmentation
This repository has two main scripts pre_train_im_impainting.py and segmentation_main_matching.py.

The pre_train_im_impainting.py trains a selected model to fill certain imprinted parts of the image. These occlusions are generated randomly with different sizes and seek to have the model learn the image details to further improve the image segmentation in future steps. 

The pre_train_im_impainting.py file has the following arguments: 
    --comet_API_KEY,
    --pat,
    --max_epoch,
    --batch_size_list,
    --learning_rate_list,
    --metric,
    --data_aug,
    --model,
    --disease_class

The comet_API_KEY signals to which workspace and project the project is being logged. If no comet_API_KEY is provided the script will save the data to a csv file in the folder logs, with the hyperparameters associated with the file, in the file name.

The pat argument signals after how many epochs should the model training stop. 

The max_epoch signals the maximum number of epochs permitted for each run. 

The batch_size_list argument signals how many batch sizes can be incorporated into the model training. 

The learning_rate_list argument signals how many learning rates can be incorporated into the model training.

The meric argument allows you to select between MSE, SSIM, FID and PNSR (between the original image and the image reconstruction) to determine which metric is selected to choose which is the best model from which epoch.

The data_aug argument can be selected between normal, gaussian_blur, h_flip, v_flip, h_and_v_flip and h_flip_blur. Each represents a data augmentation techninque that will be applied to the images on the fly. 

The model argument can be selected between FCN, U2NET, UNet_3Plus, AttentionUNet, UnetResnet34 and UNet. All these architectures can be incorporated into the trainning script. 

The disease_class can be selected between rvo_treated, others, normal and all. Each represents a different disease to use in training, while the all class joins every disease for training and testing. 

The following image shows an example of these image inpainting techniques. In the left there is the image prediction, in the middle there is an occluded image, while in the right there is the original image.

<p align="center">
    <img width="400" height="133" alt="image" src="https://github.com/user-attachments/assets/1d9d6af1-e6a4-45ae-ad02-13002c3358cd" style="float:right;"/>
</p>
The segmentation_main_matching.py has the code responsible for image segmentation training. The models are finetuned to one of the ophtalmologist's segmentation. 

The segmentation_main_matching.py file has the following arguments:
    --pat,
    --max_epoch,
    --total_fold_num,
    --batch_size_list,
    --learning_rate_list,
    --loss,
    --data_aug,
    --model,
    --disease_class,
    --weight_init,
    --oftal_2

The pat argument signals after how many epochs should the model training stop. 

The max_epoch signals the maximum number of epochs permitted for each run.

The total_fold_num signals the number of different folds used in cross validation. 

The batch_size_list argument signals how many batch sizes can be incorporated into the model training. 

The learning_rate_list argument signals how many learning rates can be incorporated into the model training.

The data_aug argument can be selected between normal, gaussian_blur, h_flip, v_flip, h_and_v_flip and h_flip_blur. Each represents a data augmentation techninque that will be applied to the images on the fly. 

The model argument can be selected between FCN, U2NET, UNet_3Plus, AttentionUNet, UnetResnet34 and UNet. All these architectures can be incorporated into the trainning script. 

The disease_class can be selected between rvo_treated, others, normal and all. Each represents a different disease to use in training, while the all class joins every disease for training and testing.

The weight_init can be normal or im_paint. If the normal part is selected the model weights will be initialized with a truncated normal distribuition, if the im_paint option is selected the model will be initialized based on the weights from the image impainting model. 

The oftal_2 option can be True or False, if True the annotations from the second oftalmologist will also be used in the image testing. Otherwise, the annotations from the second oftalmologist will not be used. 

The following image shows an example of the image segmentation results. In the left there is the image annotation, in the middle there is the segmentation prediction, while in the right there is the image prediction countour.

<p align="center">
    <img width="400" height="133" alt="image" src="https://github.com/user-attachments/assets/31a4d4be-ba2b-4896-8576-41fbfa43b8e7" />
</p>

The code follows the following structure: 
    segmentation_main_matching.py
    pre_train_im_impainting.py
    utils:
        __init__.py
        dataset.py
        lightning_module.py
        loss.py
        model.py
        pipeline.py

The two main scripts are in the main directory, while the remainder of the files with the necessary functions are in the utils folder. The dataset.py file has the dataloaders, the lightning_module.py file has the protocols used for the training and validation loops, the loss.py file has the losses used in traininig, and the model.py has the models used in the main scripts.

This code is used in the article [XXXX], with a A6000 NVIDIA GPU, 7 CPU cores, 32GB RAM, and 2TB storage. The operating system was Ubuntu 22.04, with an nvidia driver 590.48.01. The python version was 3.10.12 and has the following packages:

    comet_ml -------------------------------------- 3.57.0
    
    lightning ------------------------------------- 2.6.1
    
    cuda toolkit ---------------------------------- 13.1
    
    torch ----------------------------------------- 2.10
    
    torchvision ----------------------------------- 0.25.0 + cu13.0
    
    statsmodel ------------------------------------ 0.14.6
    
    matplotlib ------------------------------------ 3.10.8
    
    opencv-python --------------------------------- 4.13
    
    scikit-image ---------------------------------- 0.25.2
    
    scikit-learn ---------------------------------- 1.7.2
    
    seaborn --------------------------------------- 0.13.2
