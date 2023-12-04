ARG BASE_IMAGE

FROM $BASE_IMAGE

RUN git clone https://github.com/pytorch-labs/gpt-fast/
RUN pip install sentencepiece huggingface_hub
RUN pip uninstall torchtext torchdata torch torchvision torchaudio -y
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed
RUN cd gpt-fast && python setup.py install