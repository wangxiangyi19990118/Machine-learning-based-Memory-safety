import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
import model
from tensorflow.python import pywrap_tensorflow
import cv2
import glob

'''
def get_files(file_path):
    class_train = []
    label_train = []
    print(os.listdir(file_path))
    for train_class in os.listdir(file_path):
        for pic_name in os.listdir(file_path +'/'+ train_class):
            class_train.append(file_path +'/'+ train_class + '/' + pic_name)
            label_train.append(train_class)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    # class is 1 2 3 4 5
    label_list = [int(i) for i in label_list]
    return image_list, label_list


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_temp = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_temp, channels=3)
    # resize image
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch



logs_train_dir='/'

CHECK_POINT_DIR = './models'

train, train_label = get_files('dateset/train')

train_batch, train_label_batch = get_batches(train, train_label, 256, 256, 16, 20)



train_logits =model.inference(train_batch, 16, 2)

train_loss = model.losses(train_logits, train_label_batch)

train_op = model.trainning(train_loss, 0.001)

train_acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()


sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()



sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in np.arange(8001):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        if step % 10 == 0:
            print('Step %d, train loss=%.5f, train accuracy = %.4f%%' % (step, tra_loss, tra_acc))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        if (step + 1) == 8001:
            checkpoint_path = os.path.join(CHECK_POINT_DIR, './model_ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()
coord.join(threads)

'''
CHECK_POINT_DIR = './models'

def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 256, 256,3])

        logit = model.inference(image, 1, 2)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[256, 256, 3])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print(prediction)
            if max_index == 0:
                result = ('this is F rate: %.6f, result prediction is [%s]' % (
                prediction[:, 0], ','.join(str(i) for i in prediction[0])))
                isTrue=1
            else:
                result = ('this is T rate: %.6f, result prediction is [%s]' % (
                prediction[:, 1], ','.join(str(i) for i in prediction[0])))
                isTrue =0
            return result,isTrue



if __name__ == '__main__':
    
    checkpoint_path = os.path.join("./modelsave/model_ckpt-8000")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
    
    path_file = glob.glob('dataset/val/F/*')  # 获取当前文件夹下个数
    path_number = len(path_file)
    print(path_number)
    rate=0
    for i in range(path_number):
        imagefile=path_file[i]
        image = Image.open(imagefile)
        image = image.resize([256, 256])
        rgb = image.convert('RGB') #转化为3通道图像
        rgb=np.array(rgb)
        res,r=evaluate_one_image(rgb)
        rate+=r
        print(res)
    print(rate/path_number)
