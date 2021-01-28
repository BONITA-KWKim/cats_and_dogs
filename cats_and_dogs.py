"""
    Mask R-CNN
    Train on the Custom dataset.
    Copyright (c) 2018 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Kyungwon.KIM
    ------------------------------------------------------------
    Usage: import the module (see Jupyter notebooks for examples), or run from
    the command line as such:

    python3 cats_and_dogs.py train --dataset dataset/image --weights coco ₩
    --logs reulst/logs --layer all --epochs 2

    python3 cats_and_dogs.py detect --dataset dataset/image --weights coco ₩
    --subset test
"""

import os
import sys
import random
import math
import re
import time
import datetime
import numpy as np
import cv2
import json
import skimage.draw

from os import listdir
from os.path import isfile, join

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/cats_and_dogs/")

############################################################
#  Configurations
############################################################

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
    # Give the configuration a recognizable name
    NAME = "CatsAndDogs"
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # Background + (cat + dog)
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class CustomInferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
    
    def load_custom_data(self, dataset_dir, subset):
        """ Load a subset of the dataset.
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        source_name = "CatsAndDogs"
        classes = ["cat", "dog"]
        for i in range(len(classes)):
            self.add_class(source_name, i+1, classes[i])
        
        # Train or validation dataset
        assert subset in ["train", "val", "test"]
        image_dir = os.path.join(dataset_dir, subset)
        
        # load annotation
        dataset_json = [f for f in listdir(image_dir) if f.endswith(".json")]
        
        for item in dataset_json:
            #annotation = json.load(open(os.path.join(image_dir, item)))
            print("D, json path: {}".format(open(os.path.join(image_dir, item))))
            raw = json.load(open(os.path.join(image_dir, item)))
            if type(raw) is list:
                if 0 < len(raw):
                    annotation = raw[0]
                else:
                    continue
        
            if annotation['geometry']['type'] != 'Polygon':
                continue
            
            if type(annotation['geometry']['coordinates']) is not list:
                print("Error:", annotation)
                continue
            
            all_point_x = []
            all_point_y = []
            for a in annotation['geometry']['coordinates'][0]:
                all_point_x.append(int(a[0]))
                all_point_y.append(int(a[1]))
            
            polygons = {}
            polygons["all_points_x"] = all_point_x
            polygons["all_points_y"] = all_point_y
            kind = annotation["properties"]["classification"]["name"]
            # validation
            if not kind in classes:
                continue

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must
            # read the image. This is only managable since the dataset is
            # tiny.
            image_file = os.path.splitext(item)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)
            if False == os.path.exists(image_path):
                continue
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            # use file name as a unique image id
            self.add_image(source_name,
                           image_id=image_file,
                           path=image_path,
                           width=width, height=height,
                           polygons=polygons, type=kind)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
            Returns:
            masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            """
        # If not a custom dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "CatsAndDogs":
            return super(self.__class__, self).load_mask(image_id)
                
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        # If instance_count was more than one, change that parameter
        mask = np.zeros([info["height"], info["width"], 1],
                        dtype=np.uint8)

        rr, cc = skimage.draw.polygon(info['polygons']['all_points_y'],
                                        info['polygons']['all_points_x'])
        mask[rr, cc, 0] = 1

        class_ids = np.array([self.class_names.index(info["type"])])

        return mask.astype(np.bool), class_ids.astype(np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "CatsAndDogs":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom_data(args.dataset, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom_data(args.dataset, "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=int(args.epochs),
                layers=args.layers)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    '''
        image = skimage.io.imread(args.image)
        # detect function
        # input: the list of images
        # output:
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        results = model.detect([image], verbose=1)
        
        r = results[0]
        '''
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # read dataset
    dataset = CustomDataset()
    dataset.load_custom_data(dataset_dir, subset)
    dataset.prepare()
    
    exit(0)
    
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]

        for idx in range(len(r["class_ids"])):
            if 0 == idx:
                submission.append("{}, {}, {}, {}, {}, {}, {}".format(image_id,
                    indx_ids,
                    r["class_ids"][idx], round(float(r["scores"][idx]), 4),
                    r["rois"][idx],
                    dataset.image_reference(image_id).replace(ROOT_DIR, "."),
                    predicted_image_file.replace(ROOT_DIR, ".")))
                continue
            submission.append("{}, {}, {}, {}, {}, {}, {}".format("",
                indx_ids,
                r["class_ids"][idx], round(float(r["scores"][idx]), 4),
                r["rois"][idx],
                dataset.image_reference(image_id).replace(ROOT_DIR, "."),
                predicted_image_file.replace(ROOT_DIR, ".")))
    # Save to csv file
    submission = "ImageId,ground_true,class_id,scores,ROIs,Image path,Prediction\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("\n** Result Report Directory : ", submit_dir)

############################################################
#  Command Line
############################################################
if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect custom dataset.')
    parser.add_argument("command",
                     metavar="<command>",
                     help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                     metavar="/path/to/custom/dataset/",
                     help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                     metavar="/path/to/weights.h5",
                     help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                     default=DEFAULT_LOGS_DIR,
                     metavar="/path/to/logs/",
                     help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--layers', required=False,
                     default="heads",
                     metavar="heads",
                     help='Learning layers; heads or all')
    parser.add_argument('--epochs', required=False,
                     default=1,
                     metavar='test',
                     help='The number of epochs')
    parser.add_argument('--subset', required=False,
                     metavar="Dataset sub-directory",
                     help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        config = CustomInferenceConfig()
        config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
        model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_MODEL_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
                    "mrcnn_class_logits", "mrcnn_bbox_fc",
                    "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
        "Use 'train' or 'detect'".format(args.command))

