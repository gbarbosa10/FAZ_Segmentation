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
from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.loggers import CSVLogger
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_float32_matmul_precision("medium")

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--comet_API_KEY", type=str, default= True, help= "Use the comet logger for data loading, give the API_KEY to locate your project and workspace. If var is False the results will be saved to an Excel File.")
    parser.add_argument("--pat", type=int, default = 20, help = "Number of epochs after best validation result until the models stop trainning")
    parser.add_argument("--max_epoch", type=int, default = 200, help = "Maximum Limit of epochs until the models stop trainning")
    parser.add_argument("--total_fold_num", type=int, default = 10, help = "Number of folds to divide the dataset")
    parser.add_argument("--batch_size_list", default = [16, 32, 64], type = int, nargs = '*', help = "List with various batch sizes to iterate through")
    parser.add_argument("--learning_rate_list", default = [0.0001, 0.001], type = float, nargs = '*', help = "List with various learning rates to iterate through")
    parser.add_argument("--loss", default = ["MSE_Loss"], type = str, nargs = '*', help = "Loss Function available for trainning")
    parser.add_argument("--data_aug", default = ["normal"], type = str, nargs = '*', help = "Selected data augmentation method")
    parser.add_argument("--model", default = ["FCN"], type = str, nargs = '*', help = "Selected model training architecture")
    parser.add_argument("--disease_class", default = ["all"], type = str, nargs = '*', help = "Disease Class use to train the model")
    parser.add_argument("--weight_init", default = ["im_paint"], type = str, nargs = '*', help = "Specified type of weight initiation")
    parser.add_argument("--oftal_2", default = False, type = str, nargs = '*', help = "Use Seconda Ophtalmologist Annotations")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Main File")
    
    args = get_args()

    # your credentials
    # Define settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator(device='cpu')
    g.manual_seed(4)

    def count_model_weights(model):
        weights = 0
        for parameter in model.parameters():
            weights += torch.flatten(parameter.data).size()[0]
        return weights

    def reset_index_from_one(df):
        df = df.reset_index(drop=True)  # Reset index and drop old index
        df.index = range(0, len(df))  # Set new index from 1 to number of rows
        return df

    def reset_patient_from_one(df):
        #df = df.reset_index(drop=True)  # Reset index and drop old index
        df['patient'] = range(0, len(df))  # Set new index from 1 to number of rows
        return df

    def compare_models(model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if key_item_1[1].device == key_item_2[1].device  and torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    _device = f'device {key_item_1[1].device}, {key_item_2[1].device}' if key_item_1[1].device != key_item_2[1].device else ''
                    print(f'Mismtach {_device} found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly')

    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight)
        elif isinstance(module, nn.Conv2d):
            init.trunc_normal_(module.weight)

    def clean_state_dict(input):
        for key in list(input.keys()).copy():
            input[key.removeprefix('model.')] = input[key]
            del input[key]
        return input

    def ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

    num_GPU = 1

    image_size = 224
    resize_crop = torchvision.transforms.Resize((image_size, image_size))
    class_num = 2
    params = {"optimizer": "Adam", "image_resize" : image_size, "patience" : args.pat, "database" : "CHSJ", "probability function" : "log_softmax"}
    
    
    best_val_fbeta = 0
    best_val_loss = 1000
    flag = 0
    epoch = 0 
        
    images_normalization_list=False
    ensure_folder_exists("model")
    ensure_folder_exists("segmentation_models")
    mean=0
    std=1
    for loss_id in args.loss:
        for model_id in args.model:
            for data_aug_id in args.data_aug:
                for disease in args.disease_class:
                    for batch_size in args.batch_size_list:
                        for learning_rate in args.learning_rate_list:
                            for is_oftal2 in args.oftal_2:

                                if loss_id == "IoU_Loss":
                                    loss = utils.JaccardLoss()
                                if loss_id == "DiceLoss":
                                    loss = utils.DiceLoss()
                                if loss_id == "HD_Loss":
                                    loss = utils.HausdorffDTLoss()
                                if loss_id == "MSE_Loss":
                                    loss = nn.MSELoss()
                                
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

                                total_fold_num = args.total_fold_num
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

                                gss = GroupShuffleSplit(n_splits=2, train_size=.9, random_state=42)
                                gss.get_n_splits()
                                print(gss)
                                for i, (train_index, test_index) in enumerate(gss.split(train_annotations_all['filename'], train_annotations_all['label'], train_annotations_all['patient'])):
                                    print(f"Fold {i}:")
                                    print(f"  Train: index={train_index}, group={train_annotations_all.loc[train_index]}")
                                    print(f"  Test:  index={test_index}, group={train_annotations_all.loc[test_index]}")

                                    train_annotations = train_annotations_all.iloc[train_index]
                                    val_annotations = train_annotations_all.iloc[test_index]
                                    print("train label values fold: " ,  train_annotations.label.value_counts())
                                    print("val label values fold: " ,  val_annotations.label.value_counts())

                                    print("---------------------------Train---------------------------")
                                    print(len(train_annotations))

                                    print("---------------------------Validation---------------------------")
                                    print(len(val_annotations))

                                    print("---------------------------Test---------------------------")
                                    print(len(test_annotations))

                                if is_oftal2:
                                    train_dataset = utils.Dataset_bg_mask_oftal_2(train_annotations, img_dir, mean, std, 0, transform=aug)
                                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, persistent_workers=True)

                                    val_dataset = utils.Dataset_bg_mask_oftal_2(val_annotations, img_dir, mean, std, 0, transform=resize_crop)
                                    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), num_workers=5, persistent_workers=True)

                                    test_dataset = utils.Dataset_bg_mask_oftal_2(test_annotations, img_dir, mean, std, 0, transform=resize_crop)
                                    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), num_workers=5, persistent_workers=True)
                                else: 
                                    train_dataset = utils.Dataset_bg_mask(train_annotations, img_dir, mean, std, 0, transform=aug)
                                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, persistent_workers=True)

                                    val_dataset = utils.Dataset_bg_mask(val_annotations, img_dir, mean, std, 0, transform=resize_crop)
                                    val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), num_workers=5, persistent_workers=True)

                                    test_dataset = utils.Dataset_bg_mask(test_annotations, img_dir, mean, std, 0, transform=resize_crop)
                                    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), num_workers=5, persistent_workers=True)

                                if  isinstance(args.comet_API_KEY, str):
                                    logger = CometLogger(
                                    api_key=args.comet_API_KEY
                                    )

                                    logger.log_hyperparams(params)
                                    experiment_id = logger.experiment.get_key()
                                    model_name = model_id  + "_" + str(experiment_id)
                                    api_exp = logger.experiment
                                    params["lr"] = learning_rate
                                    params["loss"]= loss_id
                                    params["augmentation"] = data_aug_id
                                    params['fold'] = i
                                    params['batch_size'] = batch_size
                                    params['weight_init'] = args.weight_init 
                                    params["model"]= model_id

                                    logger.log_hyperparams(params)
                                    experiment_id = logger.experiment.get_key()
                                    model_name = model_id  + "_" + str(experiment_id)

                                else:
                                    ensure_folder_exists("logs")
                                    logger = CSVLogger("logs", name= "model_" + model_id + "_disease_" + disease + '_lr_' + str(learning_rate) + '_bs_' + str(batch_size) + '_aug_' + data_aug_id)
                                    model_name = model_id  + "_" + str(id)

                                if args.weight_init == 'normal':
                                    model.apply(initialize_weights)

                                # Pre-Train Model in Image Reconstruction
                                if args.weight_init == 'im_paint':
                                    if model_id == "FCN":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/FCN_FAZ-2197_im_inpaint.ckpt", weights_only=True)["state_dict"]))
                                    if model_id == "UNet_3Plus":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/UNet_3Plus_FAZ-2149_im_inpaint.ckpt", weights_only=True)["state_dict"]))
                                    if model_id == "U2NET":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/U2NET_FAZ-2188_im_inpaint.ckpt", weights_only=True)["state_dict"]))
                                    if model_id == "UNet":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/UNet_FAZ-2182_im_inpaint.ckpt", weights_only=True)["state_dict"]))
                                    if model_id == "UnetResnet34":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/UnetResnet34_FAZ-2155_im_inpaint.ckpt", weights_only=True)["state_dict"]))
                                    if model_id == "AttentionUNet":
                                        model.load_state_dict(clean_state_dict(torch.load("best_models/AttentionUNet_FAZ-2140_im_inpaint.ckpt", weights_only=True)["state_dict"]))

                                # Model Finetuning
                                if is_oftal2:
                                    lit_model = utils.LitModel_SimpleSemanticSegOphtal2(model, batch_size, learning_rate, "model" + model_name, "segmentation_models/", loss)
                                else: 
                                    lit_model = utils.LitModel_SimpleSemanticSeg(model, batch_size, learning_rate, "model" + model_name, "segmentation_models/", loss)
                                
                                early_stop_callback = EarlyStopping(monitor="Val Jaccard Index: ", min_delta=0.001, patience=args.pat, verbose=False, mode="max")
                                save_model = ModelCheckpoint(dirpath=model_path , filename= model_name, mode="max", monitor="Val Jaccard Index: ")

                                trainer = Trainer(max_epochs=args.max_epoch, callbacks= [early_stop_callback, save_model], accumulate_grad_batches=1, accelerator="gpu", log_every_n_steps=1, logger= logger, deterministic=True)
                                trainer.fit(lit_model, train_dataloader, val_dataloader)
                                trainer.test(dataloaders= test_dataloader, ckpt_path=model_path + model_name + ".ckpt")
                                
                                if  isinstance(args.comet_API_KEY, str):
                                    if api_exp.get_metric("Test Jaccard Index: ") <= 0.80:
                                        os.remove(model_path + model_name + ".ckpt")
                                    
                                    # Ophtalomologist 1
                                    api_exp.log_image(Image.open("model_ims/bland_altman_area_lr_" + str(learning_rate) + "_all_images_.png"), name="test/bland_altman_area_total")
                                    api_exp.log_image(Image.open("model_ims/bland_altman_perimiter_lr_" + str(learning_rate) + "_all_images_.png"), name="test/bland_altman_perimiter_total")
                                    api_exp.log_image(Image.open("model_ims/box_plot_lr_" + str(learning_rate) + "_all_images_.png"), name="test/box_plot_total")
                                    
                                    api_exp.log_image(Image.open("segmentation_models/test_batch_bg_3_image_joint_0_" + str(params["lr"]) + "_" + str(batch_size) + ".png"), name="test/test_im_0")
                                    api_exp.log_image(Image.open("segmentation_models/test_batch_bg_3_image_joint_1_" + str(params["lr"]) + "_" + str(batch_size) + ".png"), name="test/test_im_1")
                                    api_exp.log_image(Image.open("segmentation_models/test_batch_bg_3_image_joint_2_" + str(params["lr"]) + "_" + str(batch_size) + ".png"), name="test/test_im_2")
                                    api_exp.log_image(Image.open("segmentation_models/test_batch_bg_3_image_joint_3_" + str(params["lr"]) + "_" + str(batch_size) + ".png"), name="test/test_im_3")
                                    api_exp.log_image(Image.open("segmentation_models/test_batch_bg_3_image_joint_4_" + str(params["lr"]) + "_" + str(batch_size) + ".png"), name="test/test_im_4")
                                    
                                    # Ophtalomologist 2
                                    if is_oftal2:
                                        api_exp.log_image(Image.open("model_ims/bland_altman_area_lr_" + str(learning_rate) + "_all_images_ophtal_2.png"), name="test/bland_altman_area_total_ophtal_2")
                                        api_exp.log_image(Image.open("model_ims/bland_altman_perimiter_lr_" + str(learning_rate) + "_all_images_ophtal_2.png"), name="test/bland_altman_perimiter_total_ophtal_2")
                                        api_exp.log_image(Image.open("model_ims/box_plot_lr_" + str(learning_rate) + "_all_images_ophtal_2.png"), name="test/box_plot_total_ophtal_2")

                                        api_exp.log_image(Image.open("segmentation_models/test_batch_bg__3_image_joint_0_" + str(params["lr"]) + "_" + str(batch_size) + "_ophtal_2.png"), name="test/test_im_0_ophtal_2")
                                        api_exp.log_image(Image.open("segmentation_models/test_batch_bg__3_image_joint_1_" + str(params["lr"]) + "_" + str(batch_size) + "_ophtal_2.png"), name="test/test_im_1_ophtal_2")
                                        api_exp.log_image(Image.open("segmentation_models/test_batch_bg__3_image_joint_2_" + str(params["lr"]) + "_" + str(batch_size) + "_ophtal_2.png"), name="test/test_im_2_ophtal_2")
                                        api_exp.log_image(Image.open("segmentation_models/test_batch_bg__3_image_joint_3_" + str(params["lr"]) + "_" + str(batch_size) + "_ophtal_2.png"), name="test/test_im_3_ophtal_2")
                                        api_exp.log_image(Image.open("segmentation_models/test_batch_bg__3_image_joint_4_" + str(params["lr"]) + "_" + str(batch_size) + "_ophtal_2.png"), name="test/test_im_4_ophtal_2")
                                    
                                    api_exp.end()
                                del train_dataloader
                                del val_dataloader
                                del test_dataloader
                                del lit_model
                                del trainer
                                del model
                                del loss
                                gc.collect()