from keras.callbacks import EarlyStopping, ModelCheckpoint

from data import load_train_data, augment
from models import UNet
from utils import mIOU, config_keras


if __name__ == '__main__':
    config_keras()
    X_train_orig, Y_train_orig = load_train_data()
    X_train, Y_train = augment(X_train_orig, Y_train_orig)
    model = UNet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mIOU])
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=30,
                        callbacks=[earlystopper, checkpointer])


