import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import utils.loss_functions
import utils.utils
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result
def Generator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


class CycleGan(keras.Model):
    def __init__(
        self,
        qpaintings_generator,
        photo_generator,
        qpaintings_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = qpaintings_generator
        self.p_gen = photo_generator
        self.m_disc = qpaintings_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_qpaintings, real_photo = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to qpaintings back to photo
            fake_qpaintings = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_qpaintings, training=True)

            # qpaintings to photo back to qpaintings
            fake_photo = self.p_gen(real_qpaintings, training=True)
            cycled_qpaintings = self.m_gen(fake_photo, training=True)

            # generating itself
            same_qpaintings = self.m_gen(real_qpaintings, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_qpaintings = self.m_disc(real_qpaintings, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_qpaintings = self.m_disc(fake_qpaintings, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            qpaintings_gen_loss = self.gen_loss_fn(disc_fake_qpaintings)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_qpaintings, cycled_qpaintings, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_qpaintings_gen_loss = qpaintings_gen_loss + total_cycle_loss + self.identity_loss_fn(real_qpaintings, same_qpaintings, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            qpaintings_disc_loss = self.disc_loss_fn(disc_real_qpaintings, disc_fake_qpaintings)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        qpaintings_generator_gradients = tape.gradient(total_qpaintings_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        qpaintings_discriminator_gradients = tape.gradient(qpaintings_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(qpaintings_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(qpaintings_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        
        return {
            "qpaintings_gen_loss": total_qpaintings_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "qpaintings_disc_loss": qpaintings_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

class create_cyclegan_model:
    def __init__(self):
        qpaintings_generator = Generator()
        photo_generator = Generator() 

        qpaintings_discriminator = Discriminator()
        photo_discriminator = Discriminator()

        qpaintings_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        qpaintings_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        cycle_gan_model = CycleGan(
            qpaintings_generator, photo_generator, qpaintings_discriminator, photo_discriminator
        )

        cycle_gan_model.compile(
            m_gen_optimizer = qpaintings_generator_optimizer,
            p_gen_optimizer = photo_generator_optimizer,
            m_disc_optimizer = qpaintings_discriminator_optimizer,
            p_disc_optimizer = photo_discriminator_optimizer,
            gen_loss_fn = utils.loss_functions.generator_loss,
            disc_loss_fn = utils.loss_functions.discriminator_loss,
            cycle_loss_fn = utils.loss_functions.calc_cycle_loss,
            identity_loss_fn = utils.loss_functions.identity_loss
        )
        self.cycle_gan_model= cycle_gan_model
        self.qpaintings_generator=qpaintings_generator

    def fit(self,painting_ds, photo_ds,epoch_n):
        self.cycle_gan_model.fit(tf.data.Dataset.zip((painting_ds, photo_ds)),epochs=epoch_n)
    def give_model(self):
        return self.qpaintings_generator