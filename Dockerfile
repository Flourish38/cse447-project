# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
FROM allennlp/allennlp:v2.0.1-cuda10.2
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install tqdm
# RUN pip install numpy
RUN pip install nltk
# RUN pip install torch
# RUN pip install allennlp==2.0.1
RUN pip install pygtrie

# removes the entrypoint from allennlp's image
ENTRYPOINT []