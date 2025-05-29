import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Hiperparametry
EPOCHS = 100        # Liczba epok treningu
BATCH_SIZE = 64     # Rozmiar batcha
NOISE_DIM = 100     # Wymiar wektora szumu dla generatora
LEARNING_RATE_G = 0.0001  # Współczynnik uczenia dla generatora
LEARNING_RATE_D = 0.0004  # Współczynnik uczenia dla dyskryminatora
BETA1 = 0.5     # Parametr Adam optimizer
BETA2 = 0.999

# Wymiary obrazka MNIST
IMG_WIDTH = 28
IMG_HEIGHT = 28
CHANNELS = 1

# --- Ładowanie i przygotowanie danych MNIST ---
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Przetwarzanie obrazów: normalizacja do zakresu [-1, 1] i zmiana kształtu
train_images = train_images.reshape(train_images.shape[0], IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype('float32')
# Normalizacja obrazów do zakresu [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = train_images.shape[0]
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Utworzenie iteratora
iterator = dataset.make_initializable_iterator()
real_images_batch = iterator.get_next()


# --- Poprawiona definicja sieci Generatora ---
def generator(z, is_training=True):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Warstwa gęsta i reshape do 7x7x512 (więcej filtrów)
        dense = tf.layers.dense(z, 7 * 7 * 512)
        dense = tf.reshape(dense, [-1, 7, 7, 512])
        dense = tf.layers.batch_normalization(dense, training=is_training)
        dense = tf.nn.relu(dense)

        # Pierwsza warstwa dekonwolucji: 7x7x512 -> 14x14x256
        conv1 = tf.layers.conv2d_transpose(
            dense, 256, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)

        # Druga warstwa dekonwolucji: 14x14x256 -> 28x28x128
        conv2 = tf.layers.conv2d_transpose(
            conv1, 128, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)

        # Dodatkowa warstwa konwolucji dla lepszej jakości
        conv3 = tf.layers.conv2d(
            conv2, 64, (3, 3), strides=(1, 1), padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)

        # Ostatnia warstwa: 28x28x64 -> 28x28x1
        output = tf.layers.conv2d(
            conv3, CHANNELS, (3, 3), strides=(1, 1), padding='same', use_bias=False,
            activation=tf.nn.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        return output


# --- Poprawiona definicja sieci Dyskryminatora ---
def discriminator(image, reuse=False, is_training=True):
    with tf.variable_scope("discriminator", reuse=reuse):
        # Pierwsza warstwa: 28x28x1 -> 14x14x64
        conv1 = tf.layers.conv2d(
            image, 64, (4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
        conv1 = tf.layers.dropout(conv1, rate=0.3, training=is_training)

        # Druga warstwa: 14x14x64 -> 7x7x128
        conv2 = tf.layers.conv2d(
            conv1, 128, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        conv2 = tf.layers.dropout(conv2, rate=0.3, training=is_training)

        # Trzecia warstwa: 7x7x128 -> 4x4x256
        conv3 = tf.layers.conv2d(
            conv2, 256, (4, 4), strides=(2, 2), padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        conv3 = tf.layers.dropout(conv3, rate=0.3, training=is_training)

        # Spłaszczenie
        flatten = tf.layers.flatten(conv3)

        # Warstwa wyjściowa
        logits = tf.layers.dense(
            flatten, 1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        return logits


# --- Budowanie modelu GAN ---
# Placeholdery
noise_input = tf.placeholder(tf.float32, shape=[None, NOISE_DIM])
real_image_input = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, CHANNELS])
is_training_pl = tf.placeholder(tf.bool, name='is_training')

# Generowanie obrazów
fake_images = generator(noise_input, is_training=is_training_pl)

# Wyjścia dyskryminatora
logits_real = discriminator(real_image_input, reuse=False, is_training=is_training_pl)
logits_fake = discriminator(fake_images, reuse=True, is_training=is_training_pl)

# --- Funkcje strat ---
# Label smoothing dla lepszej stabilności
real_labels = tf.ones_like(logits_real) * 0.9  # Label smoothing
fake_labels = tf.zeros_like(logits_fake) + 0.1

# Straty dyskryminatora
loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=real_labels))
loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=fake_labels))
d_loss = loss_real + loss_fake

# Straty generatora
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))

# --- Optymalizatory ---
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = [op for op in update_ops if 'generator' in op.name]
d_update_ops = [op for op in update_ops if 'discriminator' in op.name]

with tf.control_dependencies(d_update_ops):
    d_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE_D,
        beta1=BETA1,
        beta2=BETA2
    ).minimize(d_loss, var_list=d_vars)

with tf.control_dependencies(g_update_ops):
    g_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE_G,
        beta1=BETA1,
        beta2=BETA2
    ).minimize(g_loss, var_list=g_vars)


# --- Funkcja do generowania i zapisywania obrazów ---
def save_generated_images(epoch, examples=16, dim=(4, 4), figsize=(10, 10)):
    # Stały seed dla porównywalności
    np.random.seed(42)
    test_noise = np.random.normal(0, 1, size=[examples, NOISE_DIM])

    # Wygenerowanie obrazów
    generated_images = sess.run(fake_images, feed_dict={noise_input: test_noise, is_training_pl: False})

    # Reskaluj obrazy z [-1, 1] do [0, 1]
    generated_images = (generated_images + 1) / 2.0

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        # Użyj interpolacji 'nearest' dla ostrzejszych obrazów
        plt.imshow(generated_images[i, :, :, 0], cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('gan_images'):
        os.makedirs('gan_images')
    plt.savefig('gan_images/mnist_gan_epoch_{:04d}.png'.format(epoch), dpi=150, bbox_inches='tight')
    plt.close()


# --- Sesja TensorFlow i pętla treningowa ---
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print("Rozpoczęcie treningu...")
    start_time = time.time()

    # Stały szum do śledzenia postępów
    fixed_noise = np.random.normal(0, 1, size=[16, NOISE_DIM])

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        sess.run(iterator.initializer)
        batch_num = 0
        g_epoch_loss = 0
        d_epoch_loss = 0

        while True:
            try:
                real_batch = sess.run(real_images_batch)
                current_batch_size = real_batch.shape[0]

                # Trenuj dyskryminator 2 razy na każdy trening generatora
                for _ in range(2):
                    noise_batch = np.random.normal(0, 1, size=[current_batch_size, NOISE_DIM])
                    _, d_batch_loss = sess.run(
                        [d_optimizer, d_loss],
                        feed_dict={real_image_input: real_batch, noise_input: noise_batch, is_training_pl: True}
                    )

                # Trenuj generator
                noise_batch = np.random.normal(0, 1, size=[current_batch_size, NOISE_DIM])
                _, g_batch_loss = sess.run(
                    [g_optimizer, g_loss],
                    feed_dict={noise_input: noise_batch, is_training_pl: True}
                )

                g_epoch_loss += g_batch_loss
                d_epoch_loss += d_batch_loss
                batch_num += 1

            except tf.errors.OutOfRangeError:
                break

        epoch_time = time.time() - epoch_start_time
        avg_g_loss = g_epoch_loss / batch_num
        avg_d_loss = d_epoch_loss / batch_num

        print(
            f"Epoka {epoch}/{EPOCHS} | Czas: {epoch_time:.2f}s | Strata G: {avg_g_loss:.4f} | Strata D: {avg_d_loss:.4f}")

        # Zapisz obrazy co 5 epok
        if epoch % 5 == 0 or epoch == 1:
            save_generated_images(epoch)

    total_time = time.time() - start_time
    print(f"\nTrening zakończony. Całkowity czas: {total_time:.2f}s")

    # Końcowe obrazy
    save_generated_images(EPOCHS, examples=25, dim=(5, 5), figsize=(12, 12))
    print("Zapisano końcowe wygenerowane obrazy w folderze 'gan_images'.")
