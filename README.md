# fakenews

#### Establishing the baseline

We will use the fever dataset from 2018. There is a newer fever dataset and shared task that is due for submission on 27th July 2021.

Someone did the hard work for us, processed the Wiki pages and created a database containing all the Wiki pages. They have also provided the TFIDF index that can be used to query documents slimilar to a text/sentence.

The pre-processed dataset, the SQLlite database and the TDIDF index is available in a container

`docker create --name fever-common feverai/common`

Use the Dockerfile to build the image we will be using to train our model.

Just like the FEVER paper, we will use a simple MLP for a baseline model. The eval dataset for the baseline model has already been provided in the feverai/common container.

`docker build -it fever .`

`docker run --rm -it --volumes-from fever-common:ro -v ~/fakenews/:/app -p 8888:8888 fever:latest`

In the container, pull the code from this repo and install the requirements.

`pip install -r requirements.txt`

Launch the nodebook

`jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --notebook-dir=/app`


