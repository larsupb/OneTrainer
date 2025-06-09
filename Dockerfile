FROM nvidia/cuda:12.6.1-devel-ubuntu22.04
LABEL authors="lars"

# Install git and curl
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh \
    && ln -s /opt/conda/bin/conda /usr/bin/conda

# Set environment variables for conda \
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false \
    CONDA_ALWAYS_YES=true \
    CONDA_DEFAULT_ENV=base

# Clone onetrain repository with branch "rest" to /app
RUN git clone https://github.com/larsupb/OneTrainer.git --branch rest /app

# Set the working directory
WORKDIR /app

RUN ./install.sh

# Install libGL,  libgthread-2.0
RUN apt-get update && \
    apt-get install -y libglib2.0-dev libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# Install additional Python packages in the conda environment
RUN conda run -p /app/conda_env pip install fastapi==0.111.0 uvicorn[standard]==0.30.1 pydantic==2.7.1 python-multipart==0.0.9 requests==2.31.0

COPY remote_srv.py /app/remote_srv.py

# Expose the port the app runs on
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

