ARG BASE_IMAGE

FROM $BASE_IMAGE

RUN cd /home/model-server/ && git clone https://github.com/pytorch-labs/gpt-fast/
RUN pip install sentencepiece huggingface_hub
RUN pip uninstall torchtext torchdata torch torchvision torchaudio torchserve -y
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed
RUN cd /home/model-server/gpt-fast/ && python setup.py install
RUN pip install pygit2
RUN cd /home/model-server/ && git clone https://github.com/pytorch/serve.git
RUN cd /home/model-server/serve && python ts_scripts/install_from_src.py
