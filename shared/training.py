import tensorflow as tf
import numpy as np
import os, json, time
import math
from tensorflow.keras.callbacks import Callback
from typing import Optional

def get_callbacks(dir_path, is_enterprise_project, max_training_time_s, max_gpu_time_s, enable_tensorboard):
    callbacks = []

    handle_training_deadline_callback = HandleTrainingDeadline(
        is_enterprise_project=is_enterprise_project, max_training_time_s=max_training_time_s,
        max_gpu_time_s=max_gpu_time_s)

    callbacks.append(handle_training_deadline_callback)

    if enable_tensorboard:
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(dir_path, 'tensorboard_logs'),
                                                     # Profile batches 1-100
                                                     profile_batch=(1,101))
        callbacks.append(tb_callback)

    return callbacks

def get_friendly_time(total_length_s):
    hours = math.floor(total_length_s / 3600)
    total_length_s -= (hours * 3600)
    minutes = math.floor(total_length_s / 60)
    total_length_s -= (minutes * 60)
    seconds = math.floor(total_length_s)

    tt = ''
    if (hours > 0):
        tt = tt + str(hours) + 'h '
    if (hours > 0 or minutes > 0):
        tt = tt + str(minutes) + 'm '
    tt = tt + str(seconds) + 's '
    return tt.strip()

def print_training_time_exceeded(is_enterprise_project, max_training_time_s, total_time):
    print('')
    print('ERR: Estimated training time (' + get_friendly_time(total_time) + ') ' +
        'is larger than compute time limit (' + get_friendly_time(max_training_time_s) + ').')
    print('')
    if (is_enterprise_project):
        print('You can up the compute time limit under **Dashboard > Performance settings**')
    else:
        print('See https://docs.edgeimpulse.com/docs/tips-and-tricks/lower-compute-time on tips to lower your compute time requirements.')
        print('')
        print('Alternatively, the enterprise version of Edge Impulse has no limits, see ' +
            'https://www.edgeimpulse.com/pricing for more information.');

def check_gpu_time_exceeded(max_gpu_time_s, total_time):
    # Check we have a limit on GPU time
    if (max_gpu_time_s == None):
        return

    # Check we're running on GPU
    device_count = ei_tensorflow.gpu.get_gpu_count()
    if (device_count == 0):
        return

    # Allow some tolerance
    tolerance = 1.2
    if (max_gpu_time_s * tolerance > total_time):
        return

    # Show an error message
    print('')
    print('ERR: Estimated training time (' + get_friendly_time(total_time) + ') ' +
        'is greater than remaining GPU compute time limit (' + get_friendly_time(max_gpu_time_s) + ').')
    print('Try switching to CPU for training, or contact sales (hello@edgeimpulse.com) to ' +
        'increase your GPU compute time limit.')
    print('')

    # End the job
    exit(1)

class HandleTrainingDeadline(Callback):
    """ Check when we run out of training time. """

    def __init__(self, max_training_time_s: float, max_gpu_time_s: float, is_enterprise_project: bool):
        self.max_training_time_s = max_training_time_s
        self.max_gpu_time_s = max_gpu_time_s
        self.is_enterprise_project = is_enterprise_project
        self.epoch_0_begin = time.time()
        self.epoch_1_begin = time.time()
        self.printed_est_time = False

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch == 0):
            self.epoch_0_begin = time.time()
        if (epoch == 1):
            self.epoch_1_begin = time.time()

    def on_epoch_end(self, epoch, logs):
        # on both epoch 0 and epoch 1 we want to estimate training time
        # if either is above the training time limit, then we exit
        if (epoch == 0 or epoch == 1):
            time_per_epoch_ms = 0
            if (epoch == 0):
                time_per_epoch_ms = float(time.time() - self.epoch_0_begin) * 1000
            elif (epoch == 1):
                time_per_epoch_ms = float(time.time() - self.epoch_1_begin) * 1000

            total_time = time_per_epoch_ms * self.params['epochs'] / 1000

            # uncomment this to debug the training time algo:
            # print('Epoch', epoch, '- time for this epoch: ' + get_friendly_time(time_per_epoch_ms / 1000) +
            #     ', estimated training time:', get_friendly_time(total_time))

            if (total_time > self.max_training_time_s * 1.2):
                print_training_time_exceeded(self.is_enterprise_project, self.max_training_time_s, total_time)
                exit(1)
            check_gpu_time_exceeded(self.max_gpu_time_s, total_time)
