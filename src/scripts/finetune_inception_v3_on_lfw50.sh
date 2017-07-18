#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inceptionv3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$HOME/models/inception_v3_lfw50

# Where the dataset is saved to.
DATASET_DIR=$HOME/datasets/lfw50

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=lfw50 \
  --dataset_dir=${DATASET_DIR}

# echo 'Sleep 10s to wait for GPU to cool down'
# sleep 60

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=lfw50 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=150000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=20 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


echo 'Sleep 10s to wait for GPU to cool down'
sleep 10

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=lfw50 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --batch_size=20

# Fine-tune all the new layers for 500 steps.
# python train_image_classifier.py \
#   --train_dir=${TRAIN_DIR}/all \
#   --dataset_name=lfw \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v3 \
#   --checkpoint_path=${TRAIN_DIR} \
#   --max_number_of_steps=500 \
#   --batch_size=5 \
#   --learning_rate=0.0001 \
#   --learning_rate_decay_type=fixed \
#   --save_interval_secs=60 \
#   --save_summaries_secs=60 \
#   --log_every_n_steps=10 \
#   --optimizer=rmsprop \
#   --weight_decay=0.00004

# echo 'sleep 10s to cool down GPU...'
# sleep 10

# Run evaluation.
# python eval_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR}/all \
#   --eval_dir=${TRAIN_DIR}/all \
#   --dataset_name=lfw \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=inception_v3 \
#   --batch_size=5
