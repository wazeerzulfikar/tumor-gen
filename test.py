from keras.models import load_model
from utils import *

epoch=160

generator_A2B = load_model('../saved_models/generator_A2B_{}.h5'.format(epoch))
generator_B2A = load_model('../saved_models/generator_B2A_{}.h5'.format(epoch))

generator_A2B.compile('adam', 'mae')
generator_B2A.compile('adam','mae')

image_size = 128

testA = load_dataset('../sample_datasets/horse2zebra/testA', image_size)/255.
# testB = load_dataset('../sample_datasets/horse2zebra/testB', image_size)

pred_B_images = []
for i in range(16):
	pred = generator_A2B.predict(np.expand_dims(testA[i],0))
	pred_B_images.append(np.hstack((testA[i], pred[0])))
plot_batch(pred_B_images)