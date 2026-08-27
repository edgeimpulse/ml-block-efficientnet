# Simple Ubuntu 24.04 base image with Python3.12 and CUDA setup already (for GPU training)
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.5

# Add other system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Copy Python requirements in and install them (--break-system-packages is required if we don't use a venv)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -r requirements.txt

# Pre-cache ImageNet weights so transfer learning also works with --network=none.
RUN python3 -c "import tensorflow as tf; [builder(include_top=False, pooling='avg', weights='imagenet') for builder in (tf.keras.applications.EfficientNetB0, tf.keras.applications.EfficientNetB1, tf.keras.applications.EfficientNetB2, tf.keras.applications.EfficientNetB3, tf.keras.applications.EfficientNetB4, tf.keras.applications.EfficientNetB5, tf.keras.applications.EfficientNetV2B0, tf.keras.applications.EfficientNetV2B1, tf.keras.applications.EfficientNetV2B2, tf.keras.applications.EfficientNetV2B3, tf.keras.applications.EfficientNetV2S, tf.keras.applications.EfficientNetV2M, tf.keras.applications.EfficientNetV2L)]"

# Copy the rest of your training scripts in
COPY . ./

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# Ensure we can output a valid Keras SavedModel (not a TF one) - so the data explorer works in Studio
ENV TF_USE_LEGACY_KERAS=1

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
