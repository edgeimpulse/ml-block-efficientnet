import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging, math, tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape, Lambda, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from conversion import convert_to_tf_lite, save_saved_model
from shared.parse_train_input import parse_train_input, parse_input_shape
import shared.training

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

dir_path = os.path.dirname(os.path.realpath(__file__))

WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', '/weights')

# Load files
parser = argparse.ArgumentParser(description='EfficientNet B0 model in Edge Impulse')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--model-size', type=str, required=False, default='b0')
parser.add_argument('--use-pretrained-weights', action='store_true')
parser.add_argument('--freeze-percentage-of-layers', type=int, required=False, default=90,
    help='% of layers that are frozen when transfer learning, only applies when --use-pretrained-weights is passed in')
parser.add_argument('--batch-size', type=int, required=False, default=16)
parser.add_argument('--last-layers', type=str, required=False, default='dense: 32, dropout: 0.1')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--early-stopping-patience', type=int, required=False, default=5)
parser.add_argument('--early-stopping-min-delta', type=float, required=False, default=0.001)
parser.add_argument('--data-augmentation', type=str, required=False)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

model_size = args.model_size
use_pretrained_weights = args.use_pretrained_weights
freeze_percentage_of_layers = args.freeze_percentage_of_layers
batch_size = args.batch_size
early_stopping = args.early_stopping
data_augmentation = args.data_augmentation

# grab train/test set and convert into TF Dataset
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

print('Training info:')
if use_pretrained_weights:
    print(f'    Model: {model_size.upper()} (using pretrained weights, freezing bottom {freeze_percentage_of_layers}% of layers)')
else:
    print(f'    Model: {model_size.upper()} (not using pretrained weights)')
print(f'    Epochs:', args.epochs)
print(f'    Learning rate:', args.learning_rate)
if early_stopping:
    print(f'    Early stopping: yes (patience: {args.early_stopping_patience}, min delta: {args.early_stopping_min_delta})')
else:
    print('    Early stopping: no')
print(f'    Last layers: {args.last_layers}')
print(f'    Data augmentation: {args.data_augmentation if args.data_augmentation else "None"}')
print(f'    Batch size:', batch_size)
print(f'    Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

# send out a progress update every interval_s seconds.
callbacks.append(shared.training.BatchLoggerCallback(
    batch_size=batch_size, train_sample_count=len(X_train), epochs=args.epochs, ensure_determinism=False))

# Saves the best model, based on validation loss (hopefully more meaningful than just accuracy)
best_model_temp_dir = tempfile.TemporaryDirectory()
BEST_MODEL_PATH = os.path.join(best_model_temp_dir.name, 'best_model.hdf5')
callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='auto',
    # It's important to save and load the whole model and not just the weights because,
    # if we do any fine tuning during transfer learning, the fine tuned model has a
    # slightly different data structure.
    save_weights_only=False,
    verbose=0))

if early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta))

if args.info_file:
    input = parse_train_input(args.info_file) if args.info_file else None

    callbacks = callbacks + shared.training.get_callbacks(
        dir_path=dir_path,
        is_enterprise_project=input.isEnterpriseProject,
        max_training_time_s=input.maxTrainingTimeSeconds,
        max_gpu_time_s=input.remainingGpuComputeTimeSeconds,
        enable_tensorboard=input.tensorboardLogging)

# model architecture
if model_size == 'b0':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb0_notop.h5')
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, pooling='avg', weights=None, classes=classes)
elif model_size == 'b1':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb1_notop.h5')
        base_model = tf.keras.applications.EfficientNetB1(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB1(include_top=False, pooling='avg', weights=None, classes=classes)
elif model_size == 'b2':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb2_notop.h5')
        base_model = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB2(include_top=False, pooling='avg', weights=None, classes=classes)
elif model_size == 'b3':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb3_notop.h5')
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, pooling='avg', weights=None, classes=classes)
elif model_size == 'b4':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb4_notop.h5')
        base_model = tf.keras.applications.EfficientNetB4(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB4(include_top=False, pooling='avg', weights=None, classes=classes)
elif model_size == 'b5':
    if use_pretrained_weights:
        weights_path = os.path.join(WEIGHTS_PREFIX, 'efficientnetb5_notop.h5')
        base_model = tf.keras.applications.EfficientNetB5(include_top=False, pooling='avg', weights=weights_path, classes=classes)
    else:
        base_model = tf.keras.applications.EfficientNetB5(include_top=False, pooling='avg', weights=None, classes=classes)
else:
    print(f'Expected --model-size to be b0, b1, b2, b3, b4 or b5 (was {model_size})')
    exit(1)

if use_pretrained_weights:
    # What percentage of the base model's layers we will fine tune
    fine_tune_from = math.ceil(len(base_model.layers) * (freeze_percentage_of_layers / 100))

    base_model.trainable = True
    # Freeze all the layers before the 'fine_tune_from' layer
    for layer in base_model.layers[0:fine_tune_from]:
        layer.trainable = False

if data_augmentation is not None and data_augmentation.strip() != '':
    # strip quotes as well, just in case people put them in
    opts = [ x.strip().strip("'").strip('"') for x in data_augmentation.split(',') ]
    for opt in opts:
        if opt != 'brightness' and opt != 'flip' and opt != 'crop':
            print(f'Failed to parse --data-augmentation, invalid value "{opt}" (valid: brightness, flip, crop)')
            exit(1)

    # Implements the data augmentation policy
    def augment_image(image, label):
        if 'flip' in opts:
            # Flips the image randomly
            image = tf.image.random_flip_left_right(image)

        if 'crop' in opts:
            # Increase the image size, then randomly crop it down to
            # the original dimensions
            resize_factor = random.uniform(1, 1.2)
            new_height = math.floor(resize_factor * MODEL_INPUT_SHAPE[0])
            new_width = math.floor(resize_factor * MODEL_INPUT_SHAPE[1])
            image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
            image = tf.image.random_crop(image, size=MODEL_INPUT_SHAPE)

        if 'brightness' in opts:
            # Vary the brightness of the image
            image = tf.image.random_brightness(image, max_delta=0.2)

        return image, label

    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

model = Sequential()
model.add(InputLayer(input_shape=MODEL_INPUT_SHAPE, name='x_input'))
model.add(Model(inputs=base_model.inputs, outputs=base_model.outputs))

if args.last_layers:
    for opt in args.last_layers.split(','):
        split = [ x.strip() for x in opt.split(':') ]
        if (len(split) != 2):
            print(f'Failed to parse --last-layers, option: "{opt}" cannot be parsed')
            exit(1)
        name, val = split
        if name == 'dense':
            try:
                val = int(val)
            except ValueError:
                print(f'Failed to parse --last-layers, option: "{opt}" value should be an int but was not')
                exit(1)
            model.add(Dense(val, activation='relu'))
        elif name == 'dropout':
            try:
                val = float(val)
            except ValueError:
                print(f'Failed to parse --last-layers, option: "{opt}" value should be a float but was not')
                exit(1)
            model.add(Dropout(val))
        else:
            print(f'Failed to parse --last-layers, option: "{opt}" key was not recognized (should be dense, dropout)')
            exit(1)

model.add(Flatten())
model.add(Dense(classes, activation='softmax'))

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)

# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
train_dataset_batch = train_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
validation_dataset_batch = validation_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

print('')
print('Training network OK')
print('')

print('Loading model with lowest validation loss...')
model = shared.training.load_best_model(BEST_MODEL_PATH)
print('Loading model with lowest validation loss OK')
print('')

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False

# Save the model to disk
save_saved_model(model, args.out_directory)

# Create tflite files (f32 / i8)
convert_to_tf_lite(model, args.out_directory, validation_dataset, MODEL_INPUT_SHAPE,
    'model.tflite', 'model_quantized_int8_io.tflite', disable_per_channel_quantization)

best_model_temp_dir.cleanup()
