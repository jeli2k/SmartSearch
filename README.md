# Prerequisites

Before starting, make sure your system meets the following requirements:

- A NVIDIA GPU that supports CUDA.
- Windows 10/11 with WSL 2 enabled.
- NVIDIA Game Ready Driver version 465.89 or higher (this is the version that supports WSL 2).

Install Docker Desktop:

Download and install Docker Desktop for Windows.
- Ensure you enable the WSL 2 feature in Docker Desktop settings:
- Right-click the Docker icon in the system tray and select Settings.
- Go to General and enable the Use the WSL 2 based engine option.
- In the Resources section, ensure that the integration with your installed WSL distributions is enabled.

Install NVIDIA Drivers:
- Download and install the latest NVIDIA driver (Game Ready Driver) that supports CUDA from NVIDIA GeForce Experience or the NVIDIA website.
- Make sure that you have a compatible GPU.

Currently NVIDIA CUDA with WSL 2 + Docker has a bug which blocks GPU access. To fix this, use the most recent Docker-Desktop Version and make following changes:
- Inside the docker-desktop folder, find the file "/etc/nvidia-container-runtime/config.toml" change **"no-cgroups"** from "true" to **"false"**

# After installing

* Start IDE & open a Terminal

* Build & Start Docker
            
        "docker-compose up --build" 
        (use "--no-cache" to build fully without cache)

This may take some time, depending on your machine + internet connection.

* Open second terminal window and use following commands to test if everything was installed correctly:

        docker-compose run --rm app python3 check_version.py
        docker-compose run --rm app python3 test_python.py
        docker-compose run --rm app python3 torch_test.py
        docker-compose run --rm app python3 test_haystack.py

# Add sourcecode

**Before using the Tool, make sure to add a Folder named "sourcecode" into root (/sourecode/) and add your codebase files into this folder.**

# Using the Tool

After building, create a new Terminal Window and start indexing & searching.

## Indexing

Make sure to specify directory to index ("sourecode" - where the sourcecode files are in)

* docker-compose run --rm app python3 codebase_search.py --index sourcecode

## Querying 

* docker-compose run --rm app python3 codebase_search.py --search "query"

Examples:

    "Where is the servername of the database?"

    "Where can I find the Coupon class?"

    "Where is the Login happening?"

    "Where are POST and GET requests handled?"

    "synonym query"

# Training

Parameters for Training need to be adjusted in train_model.py (learning rate, batch size, Epochs)

    docker-compose run --rm app python3 train/train_model.py

## Hyper-Training 
Hyper-Training uses 3 different parameters each and uses several combinations to find the best model. (Takes a long time to run!!)
    
    docker-compose run --rm app python3 train/hyper_train_model.py

# Constraints

TODO
No HTML Files

# Future Improvements

Train/Finetune Model to understand/transform all keyword conjugations to the present (e.g. verified -> verify)

# DEBUGGING 

To use python commands in terminal:

* docker-compose run app /bin/bash

Postman ElasticSearch checks:

* http://localhost:9200/codebase_index/_count
* http://localhost:9200/codebase_index/_search?q=*&size=10

SEARCH SPECIFIC FILE
* http://localhost:9200/codebase_index/_search
raw:
{
  "query": {
    "wildcard": {
      "name": "*dbaccess.php*"
    }
  },
  "size": 10
}

* docker-compose run --rm app python3 codebase_search.py --search "How are new users created?"

Cleanup: If there are any orphan containers or services:

    docker-compose down --remove-orphans


# Troubleshooting CUDA

Step-by-Step Installation of WSL 2 and NVIDIA Container Toolkit on Windows 10

## Installing WSL 2 Locally

[Installing WSL 2 - User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Install NVIDIA CUDA Toolkit
- Download link for the [CUDA Toolkit 12.4 Update 1 Download](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

- For Ubuntu: "sudo apt install nvidia-cuda-toolkit"

If the toolkit has been install under Windows AND Ubuntu:
Check if installed correctly

- nvcc --version

Currently NVIDIA CUDA with WSL 2 + Docker has a bug which blocks GPU access. To fix this, use the most recent Docker-Desktop Version and make following changes:

- Inside the file "/etc/nvidia-container-runtime/config.toml" change **"no-cgroups"** from "true" to **"false"**

- Check your Cuda version with "nvidia-smi"

- Check [NVIDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/) and use the matching CUDA Version in Dockerfile

- In this project CUDA 12.4.1 is used:

      FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

    - Validate that docker has GPU access:

          docker exec -it app ls /dev/nvidia*

          - If there is such a directory, nvidia toolkit has been installed correctly.

          docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

          - If nvidia toolkit has been installed correctly, you should see information about your GPU


If you are not using Docker Desktop use this guide by nvidia: [Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)