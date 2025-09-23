# hash:sha256:1796503d71dabb935b686ab8c82c0fb307f7ef32cb129b22de29ebb510fe2d8a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.10.0 \
        numpy==1.26.4 \
        pandas==2.1.4 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
