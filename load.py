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

# net = load_model('model-resnet50-final.h5') 匯入模型
# net = load_model('model-VGG16-final.h5')
net = load_model('model-ResNet101V2-final.h5')

cls_list = ['daisy', 'dandelion','rose','sunflower','tulip']

#進行預測
for f in range(10): 
    img = image.load_img(f_names[f], target_size=(224, 224)) 
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)   
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1]
    print(f_names[f])
    for i in top_inds:
        print('{:.3f}  {}'.format(pred[i], cls_list[i]))



