FROM continuumio/miniconda3
COPY . /app
WORKDIR /app
RUN conda env create -f conda.yaml
ENV PATH /opt/conda/envs/heart-disease-env/bin:$PATH
ENTRYPOINT ["python", "modelling.py"]