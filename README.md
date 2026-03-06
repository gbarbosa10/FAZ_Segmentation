# FAZ_Segmentation
This repository has two main scripts pre_train_im_impainting.py and segmentation_main_matching.py.

The pre_train_im_impainting.py trains a selected model to fill certain imprinted parts of the image. These occlusions are generated randomly with different sizes and seek to have the model learn the image details to further improve the image segmentation in future steps. 

The pre_train_im_impainting.py file has the following arguments: 
    --comet_API_KEY
    --pat
    --max_epoch
    --batch_size_list
    --learning_rate_list
    --metric
    --data_aug
    --model
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

<img width="292" height="97" alt="image" src="https://github.com/user-attachments/assets/1d9d6af1-e6a4-45ae-ad02-13002c3358cd" />

The segmentation_main_matching.py has the code responsible for image segmentation training. The models are finetuned to one of the ophtalmologist's segmentation. 

The segmentation_main_matching.py file has the following arguments:
    --pat
    --max_epoch
    --total_fold_num
    --batch_size_list
    --learning_rate_list
    --loss
    --data_aug
    --model
    --disease_class
    --weight_init
    --oftal_2




