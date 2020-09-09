import tensorflow as tf
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}', device_name)

