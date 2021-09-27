from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
t1 = time.time()

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATASET_PATH  = 'C:/Users/user/Desktop/學校/109-2/巨量課程2/Data/image_data/'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
BATCH_SIZE = 16
NUM_EPOCHS = 10
WEIGHTS_FINAL = 'model-resnet50-final.h5'

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# for cls, idx in train_batches.class_indices.items():
#     print('Class #{} = {}'.format(idx, cls))

rest_layer = ResNet50(include_top=False, weights='imagenet',
               input_shape=(224,224,3))

Res_model = Sequential()
Res_model.add(rest_layer)
Res_model.layers[0].trainable = False
Res_model.add(Flatten())
Res_model.add(Dropout(0.5))
Res_model.add(Dense(5, activation='softmax', name='prediction'))

Res_model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
# print(net_final.summary())
best_model_path = 'TL_best_model.h5'
checkpoint_callback = ModelCheckpoint(
    best_model_path,
    monitor='val_accuracy',
    save_best_only=True
)

history_res = Res_model.fit_generator(train_batches,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps = valid_batches.samples // BATCH_SIZE,
                            callbacks=[checkpoint_callback],
                            epochs = NUM_EPOCHS)
history_dict = history_res.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo-', label='Training loss')
plt.plot(epochs, val_loss_values, 'r--', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'b', label="Training accuracy")
plt.plot(epochs, val_acc_values, 'r', label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')

Res_model.load_weights(best_model_path) 
Res_model.save(WEIGHTS_FINAL)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import glob

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

file_path = 'C:/Users/user/Desktop/學校/109-2/巨量課程2/Data/image_data/test/'
f_names = glob.glob(file_path + '*.jpg')

net = load_model('model-resnet50-final.h5') #load model
# net = load_model('model-VGG16-final.h5')
# net = load_model('model-ResNet101V2-final.h5')

cls_list = ['daisy', 'dandelion','rose','sunflower','tulip']

#prediction
for f in range(10): 
    img = image.load_img(f_names[f], target_size=(224, 224)) 
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)   
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1]
    print(f_names[f])
    for i in top_inds:
        print('{:.3f}  {}'.format(pred[i], cls_list[i]))

