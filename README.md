

# Prerequisites

Before starting, make sure your system meets the following requirements:

- **A NVIDIA GPU that supports CUDA.**
- **Windows 10/11 with WSL 2 enabled.**
- **NVIDIA Game Ready Driver version 465.89 or higher** (this is the version that supports WSL 2).

# Add Source Code

**Before using the tool, make sure to add a folder named `sourcecode` into the root (`/sourcecode/`) and add your codebase files into this folder.**

## Install Docker Desktop:

Download and install Docker Desktop for Windows.
- Ensure you enable the WSL 2 feature in Docker Desktop settings:
  - Right-click the Docker icon in the system tray and select **Settings**.
  - Go to **General** and enable the **Use the WSL 2 based engine** option.
  - In the **Resources** section, ensure that the integration with your installed WSL distributions is enabled.

## Install NVIDIA Drivers:
- Download and install the latest NVIDIA driver (Game Ready Driver) that supports CUDA from NVIDIA GeForce Experience or the NVIDIA website.

- Make sure that you have a compatible GPU.

Currently, NVIDIA CUDA with WSL 2 + Docker has a bug which blocks GPU access. To fix this, use the most recent Docker Desktop version and make the following changes:

- Inside the `docker-desktop` folder, find the file `/etc/nvidia-container-runtime/config.toml` and change **`no-cgroups`** from `true` to **`false`**.

# After Installing

1. **Start IDE & open a Terminal**
2. **Build & Start Docker**

   ```bash
   docker-compose up --build
   ```
   *(Use `--no-cache` to build fully without cache)*

   This may take some time, depending on your machine + internet connection.

3. **Open a second terminal window and use the following commands to test if everything was installed correctly:**

   ```bash
   docker-compose run --rm app python3 tests.py
   ```

# Using the Tool

After building, create a new terminal window and start indexing & searching.

## Indexing

Make sure to specify the directory to index (`sourcecode` - where the source code files are located).

```bash
docker-compose run --rm app python3 codebase_search.py --index sourcecode
```

## Querying

```bash
docker-compose run --rm app python3 codebase_search.py --search "query"
```

Examples:

- `"Where is the server name of the database?"`
- `"Where can I find the Coupon class?"`
- `"Where is the Login happening?"`
- `"Where are POST and GET requests handled?"`

# Training

Parameters for training need to be adjusted in `train_model.py` (learning rate, batch size, epochs).

```bash
docker-compose run --rm app python3 train/train_model.py
```

## Hyper-Training (WIP)

Hyper-Training uses 3 different parameters each and several combinations to find the best model. *(Takes a long time to run!)*

```bash
docker-compose run --rm app python3 train/hyper_train_model.py
```

This still needs to be fine-tuned, and is a work in progress.

# Constraints

**TODO:** No HTML files.

# Future Improvements

Train/finetune the model to understand/transform all keyword conjugations to the present (e.g. `verified` -> `verify`).

# Debugging

To use Python commands in the terminal:

```bash
docker-compose run app /bin/bash
```

**Postman ElasticSearch checks:**

- `http://localhost:9200/codebase_index/_count`
- `http://localhost:9200/codebase_index/_search?q=*&size=10`

**Search specific file:**

- `http://localhost:9200/codebase_index/_search`

  **Raw query:**
  ```json
  {
    "query": {
      "wildcard": {
        "name": "*dbaccess.php*"
      }
    },
    "size": 10
  }
  ```

- Example command:

  ```bash
  docker-compose run --rm app python3 codebase_search.py --search "How are new users created?"
  ```

**Cleanup:** If there are any orphan containers or services:

```bash
docker-compose down --remove-orphans
```

# Troubleshooting CUDA

### Step-by-Step Installation of WSL 2 and NVIDIA Container Toolkit on Windows 10

## Installing WSL 2 Locally

[Installing WSL 2 - User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

**Install NVIDIA CUDA Toolkit:**
- Download link for the [CUDA Toolkit 12.4 Update 1](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

- For Ubuntu:

  ```bash
  sudo apt install nvidia-cuda-toolkit
  ```

**If the toolkit has been installed under both Windows and Ubuntu:**

1. **Check if installed correctly:**

   ```bash
   nvcc --version
   ```

2. **Currently, NVIDIA CUDA with WSL 2 + Docker has a bug which blocks GPU access. To fix this, use the most recent Docker Desktop version and make the following changes:**

   - Inside the file `/etc/nvidia-container-runtime/config.toml`, change **`no-cgroups`** from `true` to **`false`**.

3. **Check your CUDA version with:**

   ```bash
   nvidia-smi
   ```

4. **Check [NVIDIA Docker Hub](https://hub.docker.com/r/nvidia/cuda/) and use the matching CUDA version in `Dockerfile`.**

   - In this project, CUDA 12.4.1 is used:

     ```dockerfile
     FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
     ```

5. **Validate that Docker has GPU access:**

   ```bash
   docker exec -it app ls /dev/nvidia*
   ```

   - If there is such a directory, NVIDIA toolkit has been installed correctly.

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
   ```

   - If NVIDIA toolkit has been installed correctly, you should see information about your GPU.

**If you are not using Docker Desktop, use this guide by NVIDIA:** [Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

