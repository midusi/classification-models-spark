# A comparative study of the performance of four classification algorithms from the Apache Spark ML library

<!-- TODO: add link to article -->
This repository contains the source code of the paper published at the CACIC 2021 conference entitled "A comparative study of the performance of four classification algorithms from the Apache Spark ML library".


## Requirements

- Python 3.5 or higher.


## Installation

1. First, create a Python virtualenv: `python3 -m venv venv`
1. Activate the virtualenv: `source venv/bin/activate`
1. Secondly, install the requirements: `pip install -r requirements.txt`


## Execution

1. All the experiments are computed from the file `main.py`. Maybe you want to edit some parameters (there are some *TODO* sentences to check before running the script). To run the script **locally**: `python3 main.py`
1. To run **in a distributed environment** (i.e. in an Apache Spark cluster), we use the configuration provided by [this repository][jware-spark]. Once the Spark cluster is ready, just run the following command to run the benchmarks detailed in the article:
    1. Enter the master node: `docker container exec -it [master node container ID] bash`
    1. Once inside the container: `spark-submit main.py &> output_and_errors.txt &`

That command will run the experiments in background redirecting the stdout and stderr to file `output_and_errors.txt`.


<!-- TODO: complete this section when article is published -->
<!-- ## Considerations

If you use any part of our code, or SGDNet is useful for your research, please consider citing:

@inproceedings{camele2021spark,
  title={},
  author={},
  booktitle={},
  year={2021},
  organization={}
} -->

[jware-spark]: https://github.com/jware-solutions/docker-big-data-cluster
