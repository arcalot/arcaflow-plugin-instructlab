# Package path for this plugin module relative to the repo root
ARG package=arcaflow_plugin_instructlab

# Base that supports InstructLab
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubi9 as os_base
RUN dnf -y install python3.11-pip rust cargo && ln -s /usr/bin/python3.11 /usr/bin/python
RUN dnf install -y python3.11 python3.11-devel git python3-pip make automake gcc gcc-c++
RUN python3.11 -m ensurepip
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && dnf repolist && dnf config-manager --set-enabled cuda-rhel9-x86_64 && dnf config-manager --set-enabled cuda && dnf config-manager --set-enabled epel && dnf update -y
RUN dnf install -y libcudnn8 nvidia-driver-NVML nvidia-driver-cuda-libs
RUN python3.11 -m pip install --force-reinstall nvidia-cuda-nvcc-cu12
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" \
    && export CUDA_HOME=/usr/local/cuda \
    && export PATH="/usr/local/cuda/bin:$PATH" \
    && export XLA_TARGET=cuda120 \
    && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
WORKDIR /app

FROM os_base AS base
RUN dnf -y install gcc-c++ python3.11-devel openssl-devel \
 && dnf clean all

# Setup poetry
RUN python3.11 -m pip install requests>=2.32.0 poetry==1.4.2 \
 && python3.11 -m poetry config virtualenvs.create false

# Setup coverage
RUN python3.11 -m pip install coverage==7.2.7 \
 && mkdir /htmlcov


# STAGE 1 -- Build module dependencies and run tests
# The 'poetry' and 'coverage' modules are installed and verson-controlled in the
# quay.io/arcalot/arcaflow-plugin-baseimage-python-buildbase image to limit drift
FROM base as build
ARG package

COPY poetry.lock /app/
COPY pyproject.toml /app/

# Convert the dependencies from poetry to a static requirements.txt file
RUN python -m poetry install --without dev --no-root \
 && python -m poetry export -f requirements.txt --output requirements.txt --without-hashes

COPY ${package}/ /app/${package}
COPY tests /app/${package}/tests

ENV PYTHONPATH /app/${package}
WORKDIR /app/${package}

# Run tests and return coverage analysis
RUN python -m coverage run tests/test_${package}.py \
 && python -m coverage html -d /htmlcov --omit=/usr/local/*


# STAGE 2 -- Build final plugin image
FROM base
ARG package

COPY --from=build /app/requirements.txt /app/
COPY --from=build /htmlcov /htmlcov/
COPY LICENSE /app/
COPY README.md /app/
COPY ${package}/ /app/${package}

# Install all plugin dependencies from the generated requirements.txt file
RUN python -m pip install -r requirements.txt

WORKDIR /app/${package}

ENTRYPOINT ["python", "instructlab_plugin.py"]
CMD []

LABEL org.opencontainers.image.source="https://github.com/arcalot/arcaflow-plugin-instructlab"
LABEL org.opencontainers.image.licenses="Apache-2.0+GPL-2.0-only"
LABEL org.opencontainers.image.vendor="Arcalot project"
LABEL org.opencontainers.image.authors="Arcalot contributors"
LABEL org.opencontainers.image.title="Instructlab Plugin"
LABEL io.github.arcalot.arcaflow.plugin.version="1"
