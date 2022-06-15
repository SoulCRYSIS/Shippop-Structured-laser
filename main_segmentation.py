from __future__ import print_function
import os
from constant import BATCH_SIZE, EPOCH_SIZE, MODEL_NAME
from src.utils.HED_data_parser import DataParser
from src.networks.hed import hed
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
# import pdb


def generate_minibatches(dataParser, train=True):
    # pdb.set_trace()
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size_train*2)
        ims, ems, _ = dataParser.get_batch(batch_ids)
        yield(ims, [ems, ems, ems, ems])

######
def trainModel():
    # params
    model_name = MODEL_NAME
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')

    batch_size_train = BATCH_SIZE

    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    dataParser = DataParser(batch_size_train)

    # model
    model = hed()
    plot_model(model, to_file=os.path.join(model_dir, 'model.png'), show_shapes=True)

    # training
    # call backs
    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1, save_best_only=True)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=False, separator=',')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0,
                                        write_graph=True, write_images=False)
    
    train_history = model.fit(
                        generate_minibatches(dataParser,),
                        # max_q_size=40, workers=1,
                        steps_per_epoch=dataParser.steps_per_epoch,  #batch size
                        epochs=EPOCH_SIZE,
                        validation_data=generate_minibatches(dataParser, train=False),
                        validation_steps=dataParser.validation_steps,
                        callbacks=[checkpointer, csv_logger, tensorboard])

    # pdb.set_trace()

    print(train_history)