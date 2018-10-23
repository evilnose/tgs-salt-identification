import argparse
import os

from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam

from trainer.constants import IM_SHAPE_BBOX
from trainer.data import load_train_data, bounding_box, normalize_bounding_box, augment
from trainer.models import YoloReduced, CustomCNN1, HigherResCNN
from trainer.utils import copy_file_to_gcs, bbox_loss_batch, bbox_mIOU, binary_mse, coord_only_mIOU, ModelSave

BEST_MODEL_PATH = 'best_model.h5'
CUSTOM_MODEL = 'custom_cnn_model.h5'

if __name__ == '__main__':
    # config_keras()
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='Output directory for this job')
    parser.add_argument('--train-file',
                        required=True,
                        type=str,
                        help='Training data file (.npz) from a \'images\' and \'masks\' dict')
    parser.add_argument('--batch-size',
                        required=True,
                        type=int,
                        help='Batch size for training')
    parser.add_argument('--num-epochs',
                        required=True,
                        type=int,
                        help='Maximum number of epochs to train')
    parser.add_argument('--lambda-obj',
                        required=True,
                        type=float,
                        help='Loss function hyperparameter lambda_obj. Increase to penalize false negatives')
    parser.add_argument('--lambda-noobj',
                        required=True,
                        type=float,
                        help='Loss function hyperparameter lambda_noobj. Increase to penalize false positives')
    parser.add_argument('--lambda-coord',
                        required=True,
                        type=float,
                        help='Loss function hyperparameter lambda_coord. Increase to penalize coordinate prediction'
                             'errors')
    parser.add_argument('--lr',
                        required=True,
                        type=float,
                        help='Learning rate of Adam optimizer')
    parser.add_argument('--valid-split',
                        required=False,
                        type=float,
                        default=0.1,
                        help='Portion of validation data. Default to 0.1')
    parser.add_argument('--verbose',
                        required=False,
                        type=int,
                        default=1,
                        help='General verbosity')

    args = parser.parse_args()
    train_file = args.train_file
    job_dir = args.job_dir

    save_path = CUSTOM_MODEL
    if not job_dir.startswith('gs://'):
        save_path = os.path.join(job_dir, CUSTOM_MODEL)

    X_train, Y_train_orig = load_train_data(train_file, IM_SHAPE_BBOX)
    X_train, Y_train_orig = augment(X_train, Y_train_orig)
    Y_train = bounding_box(Y_train_orig)
    Y_train = normalize_bounding_box(Y_train)

    model = HigherResCNN()
    # model.summary()
    earlystopper = EarlyStopping(patience=4, verbose=1)
    checkpointer = ModelSave(CUSTOM_MODEL, job_dir, save_best_only=True, verbose=1)
    tblog = TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)
    adam_opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam_opt, loss=bbox_loss_batch(coord_scale=args.lambda_coord, obj_scale=args.lambda_obj,
                                                           noobj_scale=args.lambda_noobj),
                  metrics=[bbox_mIOU, binary_mse, coord_only_mIOU])
    model.fit(x=X_train, y=Y_train, batch_size=args.batch_size, epochs=args.num_epochs,
              validation_split=args.valid_split,
              callbacks=[earlystopper, checkpointer, tblog], verbose=args.verbose)

    if job_dir.startswith("gs://"):
        model.save(save_path)
        copy_file_to_gcs(job_dir, save_path)
    else:
        model.save(os.path.join(job_dir, save_path))

    #######################################################################################

    # def train_unet():
    #     X_train, Y_train = load_train_data(train_file)
    #     X_train, Y_train = augment(X_train, Y_train)
    #     model = UNet()
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mIOU])
    #     earlystopper = EarlyStopping(patience=5, verbose=1)
    #     checkpointer = ModelCheckpoint(BEST_MODEL_PATH, verbose=args.verbose, save_best_only=True)
    #     tblog = TensorBoard(
    #         log_dir=os.path.join(job_dir, 'logs'),
    #         histogram_freq=0,
    #         write_graph=True,
    #         embeddings_freq=0)
    #     model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
    #               callbacks=[earlystopper, checkpointer, tblog], verbose=2)
    #     model.save(BEST_MODEL_PATH)
    #     copy_file_to_gcs(job_dir, BEST_MODEL_PATH)
