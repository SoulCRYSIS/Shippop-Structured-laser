import os
from time import sleep
from constant import DATA_FOLDER, INPUT_IMAGE_SIZE, MODEL_NAME
from src.utils.HED_data_parser import DataParser
from src.networks.hed import hed
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image
import cv2


def predictResults():
    # environment
    K.set_image_data_format("channels_last")
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # if not os.path.isdir(model_dir): os.makedirs(model_dir)
    # model
    model = hed()
    # plot_model(model, to_file=os.path.join(model_dir, 'model.pdf'), show_shapes=True)

    # training
    # call backs
    resultFiles = os.listdir("train_results")
    print("--- select weight data ---", flush=True)
    for index, resultFile in enumerate(resultFiles):
        print(index, resultFile, flush=True)
    inputIndex = input("Which result will use for prediction? ")
    if not inputIndex.isdecimal() or int(inputIndex) >= len(resultFiles) or int(inputIndex) < 0:
        print("!!! Invalid input index !!!")
        return
    selectedResultFile = resultFiles[int(inputIndex)]
    model.load_weights("train_results/" + selectedResultFile)
    # train_history = model.predict()
    test = glob.glob("test/input/in_train_data/*")
    for image in test:
        name = image.split("\\")[-1]
        x_batch = []
        im = Image.open(image)
        (h, w) = im.size
        print("in_train_data/" + name, h, w)
        im = im.resize(INPUT_IMAGE_SIZE)
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32)
        prediction = model.predict(x_batch)
        mask = np.zeros_like(im[:, :, 0])
        for i in range(len(prediction)):
            mask += np.reshape(prediction[i], INPUT_IMAGE_SIZE)
        ret, mask = cv2.threshold(mask, np.mean(mask) + 1.2 * np.std(mask), 255, cv2.THRESH_BINARY)
        out_mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("test/output/in_train_data/" + name, out_mask)

    test = glob.glob("test/input/not_train_data/*")
    for image in test:
        name = image.split("\\")[-1]
        x_batch = []
        im = Image.open(image)
        (h, w) = im.size
        print("not_train_data/" + name, h, w)
        im = im.resize(INPUT_IMAGE_SIZE)
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32)
        prediction = model.predict(x_batch)
        mask = np.zeros_like(im[:, :, 0])
        for i in range(len(prediction)):
            mask += np.reshape(prediction[i], INPUT_IMAGE_SIZE)
        ret, mask = cv2.threshold(mask, np.mean(mask) + 1.2 * np.std(mask), 255, cv2.THRESH_BINARY)
        out_mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("test/output/not_train_data/" + name, out_mask)


if __name__ == "__main__":
    predictResults()

