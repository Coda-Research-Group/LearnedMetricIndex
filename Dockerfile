FROM continuumio/miniconda3:24.5.0-0

RUN conda init bash && \
    . /root/.bashrc && \
    conda create -n lmi -y python=3.11 && \
    conda activate lmi && \
    conda install -c pytorch -y faiss-cpu=1.8.0 && \
    conda install h5py=3.11.0 && \
    pip install --no-cache-dir numpy==1.26.4 tqdm==4.66.4 loguru==0.7.2 scikit-learn==1.5.1 && \
    pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    echo 'conda activate lmi' >> /root/.bashrc

WORKDIR /app
COPY . .

ENTRYPOINT ["/bin/bash", "-l", "-c" ]
