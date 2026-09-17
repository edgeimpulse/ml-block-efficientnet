# Simple Ubuntu 24.04 base image with Python3.12 and CUDA setup already (for GPU training)
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.5

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# Ensure we can output a valid Keras SavedModel (not a TF one) - so the data explorer works in Studio
ENV TF_USE_LEGACY_KERAS=1

# Add other system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Download weights, mirrored from https://github.com/Runist/image-classifier-keras/releases
RUN mkdir -p /weights && \
    cd /weights && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb0_notop.h5 && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb1_notop.h5 && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb2_notop.h5 && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb3_notop.h5 && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb4_notop.h5 && \
    wget https://cdn.edgeimpulse.com/pretrained-weights/efficientnet/efficientnetb5_notop.h5

# Copy Python requirements in and install them (--break-system-packages is required if we don't use a venv)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
