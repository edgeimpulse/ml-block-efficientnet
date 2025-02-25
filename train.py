import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape, Lambda, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from conversion import convert_to_tf_lite, save_saved_model

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', '/weights')

# Load files
parser = argparse.ArgumentParser(description='EfficientNet B0 model in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--model-size', type=str, required=False, default='b0')
parser.add_argument('--use-pretrained-weights', action='store_true')
parser.add_argument('--freeze-percentage-of-layers', type=int, required=False, default=90,
    help='% of layers that are frozen when transfer learning, only applies when --use-pretrained-weights is passed in')
parser.add_argument('--batch-size', type=int, required=False, default=16)
parser.add_argument('--last-layer-neurons', type=int, required=False, default=16)
parser.add_argument('--last-layer-dropout', type=float, required=False, default=0.1)
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--early-stopping-patience', type=int, required=False, default=5)
parser.add_argument('--early-stopping-min-delta', type=float, required=False, default=0.001)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

model_size = args.model_size
use_pretrained_weights = args.use_pretrained_weights
freeze_percentage_of_layers = args.freeze_percentage_of_layers
batch_size = args.batch_size
last_layer_neurons = args.last_layer_neurons
last_layer_dropout = args.last_layer_dropout
early_stopping = args.early_stopping

# grab train/test set and convert into TF Dataset
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# Weights file
dir_path = os.path.dirname(os.path.realpath(__file__))

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

if early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta))

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

model = Sequential()
model.add(InputLayer(input_shape=MODEL_INPUT_SHAPE, name='x_input'))
model.add(Model(inputs=base_model.inputs, outputs=base_model.outputs))
if last_layer_neurons > 0:
    model.add(Dense(last_layer_neurons, activation='relu'))
if last_layer_dropout > 0:
    model.add(Dropout(last_layer_dropout))
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)

# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
train_dataset_batch = train_dataset.batch(batch_size, drop_remainder=False)
validation_dataset_batch = validation_dataset.batch(batch_size, drop_remainder=False)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

print('')
print('Training network OK')
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
