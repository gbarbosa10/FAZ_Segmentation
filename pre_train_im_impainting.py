# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:20:41 2023

@author: gbarbosa
"""
import comet_ml
import utils
import torch
from torch.utils.data import DataLoader
import gc
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torchvision
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import v2
import argparse
import os
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.loggers import CSVLogger
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_float32_matmul_precision("medium")

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--comet_API_KEY", type=str, default= True, help= "Use the comet logger for data loading, give the API_KEY to locate your project and workspace. If var is False the results will be saved to an Excel File.")
    parser.add_argument("--pat", type=int, default = 30, help = "Number of epochs after best validation result until the models stop trainning")
    parser.add_argument("--max_epoch", type=int, default = 200, help = "Maximum Limit of epochs until the models stop trainning")
    parser.add_argument("--batch_size_list", default = [16, 32, 64], type = int, nargs = '*', help = "List with various batch sizes to iterate through")
    parser.add_argument("--learning_rate_list", default = [0.0001, 0.001], type = float, nargs = '*', help = "List with various learning rates to iterate through")
    parser.add_argument("--metric", default = ["FID"], type = str, nargs = '*', help = "Metric used to monitor image matching")
    parser.add_argument("--data_aug", default = ["normal"], type = str, nargs = '*', help = "Selected data augmentation method")
    parser.add_argument("--model", default = ["FCN"], type = str, nargs = '*', help = "Selected model training architecture")
    parser.add_argument("--disease_class", default = ["all"], type = str, nargs = '*', help = "Disease Class use to train the model")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Main File")
    
    args = get_args()

    # your credentials
    # Define settings for reproducibility 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator(device='cpu')
    g.manual_seed(4)

    def reset_index_from_one(df):
        df = df.reset_index(drop=True)  # Reset index and drop old index
        df.index = range(0, len(df))  # Set new index from 1 to number of rows
        return df

    def reset_patient_from_one(df):
        df['patient'] = range(0, len(df))  # Set new index from 1 to number of rows
        return df

    img_dir = "FAZ_Segmentation/"
    model_path = "model/"
    mean=0
    std=1
    image_size = 224
    id = 0
    for model_id in args.model:
        for data_aug_id in args.data_aug:
            for disease in args.disease_class:
                for batch_size in args.batch_size_list:
                    for learning_rate in args.learning_rate_list:
                            params = {"optimizer": "Adam", "image_resize" : image_size, "plateu" : "static", "database" : "sonia_extended"}

                            params["lr"] = learning_rate
                            params["model"]= model_id
                            params["augmentation"] = data_aug_id
                            params['batch_size'] = batch_size
                            params['weight_init'] = "only_im_paint"
                            
                            if model_id == "FCN": 
                                model = utils.FCN()
                            if model_id == "U2NET": 
                                model = utils.U2NET()
                            if model_id == "UNet_3Plus": 
                                model = utils.UNet_3Plus(in_channels=3,channels=64,n_classes=1)
                            if model_id == "AttentionUNet": 
                                model = utils.AttentionUNet(in_channels=3, out_channels=1)
                            if model_id == "UnetResnet34": 
                                model = utils.UnetResnet34()
                            if model_id == "UNet": 
                                model = utils.UNet(in_channels=3,channels=64, n_classes=1)

                            resize_crop = torchvision.transforms.Resize((image_size, image_size))
                            if data_aug_id == "normal":
                                aug = resize_crop
                            if data_aug_id == "gaussian_blur":
                                aug = v2.Compose([torchvision.transforms.RandomApply(transforms=[v2.GaussianBlur((5,5), 0.6)], p=0.3), resize_crop])
                            if data_aug_id == "h_flip":
                                aug = v2.Compose([torchvision.transforms.RandomApply(transforms=[v2.RandomHorizontalFlip(p=1)], p=0.3), resize_crop])
                            if data_aug_id == "v_flip":
                                aug = v2.Compose([torchvision.transforms.RandomApply(transforms=[v2.RandomVerticalFlip(p=1)], p=0.3), resize_crop])
                            if data_aug_id == "h_and_v_flip":
                                aug = v2.Compose([torchvision.transforms.RandomApply(transforms=[v2.RandomHorizontalFlip(p=1)], p=0.3), torchvision.transforms.RandomApply(transforms=[v2.RandomVerticalFlip(p=1)], p=0.3), resize_crop])
                            if data_aug_id == "h_flip_blur":
                                aug = v2.Compose([v2.Compose([torchvision.transforms.RandomApply(transforms=[v2.RandomHorizontalFlip(p=1)], p=0.3), torchvision.transforms.RandomApply(transforms=[v2.GaussianBlur((5,5), (0.1, 0.6))], p=0.3)]), resize_crop])

                            if disease == "rvo_treated":
                                img_dir = "FAZ_Segmentation/RVO Tratado/"
                            if disease == "others":
                                img_dir = "FAZ_Segmentation/Outros/"
                            if disease == "normal":
                                img_dir = "FAZ_Segmentation/Normais/"
                            if disease == "all": 
                                img_dir = "FAZ_Segmentation/"

                            model_path = "model/"
                            if disease == "rvo_treated":
                                train_annotations_all = pd.read_csv("new_csv/train_superficial_rvo_tratado.csv")
                            if disease == "others":
                                train_annotations_all = pd.read_csv("new_csv/train_superficial_outros.csv")
                            if disease == "normal":
                                train_annotations_all = pd.read_csv("new_csv/train_superficial.csv")
                            if disease == "all":
                                train_annotations_all = pd.concat([pd.read_csv("new_csv/train_superficial.csv"), pd.read_csv("new_csv/train_superficial_outros.csv"), pd.read_csv("new_csv/train_superficial_rvo_tratado.csv")], axis=0, ignore_index=True)
                            train_annotations_all = reset_index_from_one(train_annotations_all)
                            train_annotations_all = reset_patient_from_one(train_annotations_all)
                            train_annotations_all = train_annotations_all.drop("Unnamed: 0",axis=1)
                            train_annotations_all = train_annotations_all.drop("Column1;filename;label;patient;label_2",axis=1).dropna().reset_index(drop=True)
                            
                            if disease == "rvo_treated":
                                test_annotations = pd.read_csv("new_csv/test_superficial_rvo_tratado.csv")
                            if disease == "others":
                                test_annotations = pd.read_csv("new_csv/test_superficial_outros.csv")
                            if disease == "normal":
                                test_annotations = pd.read_csv("new_csv/test_superficial.csv")
                            if disease == "all":
                                test_annotations = pd.concat([pd.read_csv("new_csv/test_superficial.csv"), pd.read_csv("new_csv/test_superficial_outros.csv"), pd.read_csv("new_csv/test_superficial_rvo_tratado.csv")], axis=0, ignore_index=True)
                            
                            test_annotations = reset_index_from_one(test_annotations)
                            test_annotations = reset_patient_from_one(test_annotations)
                            test_annotations = test_annotations.drop("Unnamed: 0", axis=1).dropna()

                            pre_train_dataset = utils.Dataset_pre_train(train_annotations_all, img_dir, mean, std, 0, transform=aug)
                            pre_train_dataloader = DataLoader(pre_train_dataset, batch_size=batch_size, num_workers=5, persistent_workers=True)

                            if  isinstance(args.comet_API_KEY, str):
                                logger = CometLogger(
                                api_key=args.comet_API_KEY
                                )

                                logger.log_hyperparams(params)
                                experiment_id = logger.experiment.get_key()
                                model_name = model_id  + "_" + str(experiment_id)
                                api_exp = logger.experiment
                                api_exp.log_metric("MSE_mask_impaint: ", 10**12)
                                api_exp.log_metric("SSIM: ", 0)
                                api_exp.log_metric("FID: ", 10**12)
                                api_exp.log_metric("Peak Signal Noise Ratio: ", 0)
                                api_exp.log_metric("debug_init_loss: ", 10**12)
                                api_exp.log_metric("epoch", 0)
                                api_exp.log_metric("Pre_Trained", False)

                            else: 
                                logger = CSVLogger("logs", name= "pre_train_model_" + model_id + "_disease_" + disease + '_lr_' + str(learning_rate) + '_bs_' + str(batch_size) + '_aug_' + data_aug_id)
                                model_name = model_id  + "_" + str(id)

                            pre_train = utils.PreTrain_ImReconstruct(model, batch_size, learning_rate, "image_impainting_models/")
                            save_model = ModelCheckpoint(dirpath=model_path , filename= model_name + "_im_inpaint", mode="min", monitor="MSE_mask_impaint: ")
                            early_stop_callback = EarlyStopping(monitor=args.metric[0] + ": ", min_delta=0.0001, patience=args.pat, verbose=False, mode="min")
    
                            pre_trainer = Trainer(max_epochs=args.max_epoch,log_every_n_steps=1, logger=logger, callbacks= [early_stop_callback, save_model])
                            pre_trainer.fit(pre_train, pre_train_dataloader)

                            if  isinstance(args.comet_API_KEY, str):

                                if args.metric[0] == 'MSE':
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_MSE_ims__2_image_joint_"+ str(0) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_MSE_im_0")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_MSE_ims__2_image_joint_"+ str(1) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_MSE_im_1")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_MSE_ims__2_image_joint_"+ str(2) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_MSE_im_2")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_MSE_ims__2_image_joint_"+ str(3) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_MSE_im_3")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_MSE_ims__2_image_joint_"+ str(4) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_MSE_im_4")

                                if args.metric[0] == 'SSIM':
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_SSIM_ims__2_image_joint_"+ str(0) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_SSIM_im_0")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_SSIM_ims__2_image_joint_"+ str(1) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_SSIM_im_1")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_SSIM_ims__2_image_joint_"+ str(2) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_SSIM_im_2")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_SSIM_ims__2_image_joint_"+ str(3) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_SSIM_im_3")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_SSIM_ims__2_image_joint_"+ str(4) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_SSIM_im_4")

                                if args.metric[0] == 'FID':   
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_FID_ims__2_image_joint_"+ str(0) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_FID_im_0")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_FID_ims__2_image_joint_"+ str(1) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_FID_im_1")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_FID_ims__2_image_joint_"+ str(2) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_FID_im_2")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_FID_ims__2_image_joint_"+ str(3) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_FID_im_3")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_FID_ims__2_image_joint_"+ str(4) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_FID_im_4")

                                if args.metric[0] == 'PNSR': 
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(0) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_0")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(1) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_1")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(2) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_2")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(3) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_3")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(4) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_4")
                                    api_exp.log_image(Image.open("segmentation_models/image_impaint/pre_train_PNSR_ims__2_image_joint_"+ str(5) + "_" + str(learning_rate) + "_" + str(batch_size) + ".png"), name="test_PNSR_im_5")

                                id = id + 1
                                api_exp.end()
                            gc.collect()