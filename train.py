# # from __future__ import print_function
import argparse
from tqdm import tqdm
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from utils.lr_scheduler import get_lr_scheduler
from model.model_builder import get_model
from generator.generator_builder import get_generator
import sys
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using snapmix .')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--start-val-epoch', default=100, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--dataset', default='custom', type=str, help="choices=['cub','cars','custom']")
    parser.add_argument('--dataset-dir', default='dataset/cat_dog', type=str, help="choices=['dataset/cub','dataset/cars','custom_dataset_dir']")
    parser.add_argument('--augment', default='snapmix', type=str, help="choices=['baseline','cutmix','snapmix']")
    parser.add_argument('--model', default='ResNet50', type=str, help="choices=['ResNet50','ResNet101','EfficientNetB0']")
    parser.add_argument('--pretrain', default='imagenet', help="choices=[None,'imagenet','resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5']")
    parser.add_argument('--concat-max-and-average-pool', default=False, type=bool,help="Use concat_max_and_average_pool layer in model")
    parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[80, 150, 180], type=int)
    parser.add_argument('--warmup-lr', default=1e-4, type=float)
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='sgd', help="choices=['adam','sgd']")
    return parser.parse_args(args)

def main(args):

    train_generator, val_generator = get_generator(args)
    model = get_model(args, train_generator.num_class)
    train_generator.set_model(model.keras_model)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(args.init_lr,momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam(args.init_lr)

    lr_scheduler = get_lr_scheduler(args)
    best_val_loss = float('inf')
    best_val_acc = -1
    best_val_epoch = -1
    for epoch in range(args.epochs):
        lr = lr_scheduler(epoch)
        optimizer.learning_rate.assign(lr)

        # training
        train_loss = 0.
        train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
        for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:

            batch_imgs = model.preprocess(batch_imgs)
            with tf.GradientTape() as tape:
                logits = model.keras_model(batch_imgs, training=True)
                data_loss = loss_object(batch_labels, logits)
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.keras_model.trainable_variables
                                                        if 'bn' not in v.name])
                total_loss = data_loss + args.weight_decay * l2_loss
            grads = tape.gradient(total_loss, model.keras_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.keras_model.trainable_variables))
            train_loss += total_loss
            train_generator_tqdm.set_description(
                "epoch:{}/{},train_loss:{:.4f},lr:{:.6f}".format(epoch, args.epochs,
                                                                                 train_loss/((batch_index+1) * train_generator.batch_size),
                                                                                 optimizer.learning_rate.numpy()))

        train_generator.on_epoch_end()

        # validation
        if epoch > args.start_val_epoch:
            val_loss = 0.
            val_acc = 0.
            val_generator_tqdm = tqdm(enumerate(val_generator), total=len(val_generator))
            for batch_index, (batch_imgs, batch_labels) in val_generator_tqdm:
                batch_imgs = model.preprocess(batch_imgs)
                logits = model.keras_model(batch_imgs, training=False)
                loss_value = loss_object(batch_labels, logits)
                val_loss += loss_value
                val_true_num = tf.reduce_sum(
                    tf.cast(tf.equal(tf.argmax(batch_labels, axis=-1), tf.argmax(logits, axis=-1)),
                            tf.dtypes.float32))
                val_acc += val_true_num
                val_generator_tqdm.set_description(
                    "epoch:{},val_loss:{:.4f}".format(epoch, loss_value))
            val_loss /= len(val_generator)
            val_acc /= (len(val_generator) * val_generator.batch_size)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_epoch = epoch
            logging.info("best_epoch:{},best_val_loss:{},best_val_acc:{}".format(best_val_epoch, best_val_loss, best_val_acc))

if __name__== "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
