import tensorflow as tf
import os

def create_ds(path_p,path_p_):
    IMAGE_SIZE = [256, 256]

    def decode_image(path_,filename):
        image_string = tf.io.read_file(path_+filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [*IMAGE_SIZE, 3])
        return image

    painting_jpg_files=tf.constant(os.listdir(path_p_))
    photo_jpg_files=tf.constant(os.listdir(path_p))

    dataset_painting = tf.data.Dataset.from_tensor_slices(painting_jpg_files)
    dataset_painting = dataset_painting.map(decode_image).batch(1)


    dataset_photo = tf.data.Dataset.from_tensor_slices(photo_jpg_files)
    dataset_photo = dataset_photo.map(decode_image).batch(1)

    return dataset_painting,dataset_photo