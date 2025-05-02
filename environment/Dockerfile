# hash:sha256:820b419772d6958fb49e011632bbc78b727f584f8a0640f2fa42f5051a169a5b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.10.0 \
        numpy==1.26.4 \
        pandas==2.1.4 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
