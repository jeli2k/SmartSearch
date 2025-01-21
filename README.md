<h1><b>SmartSearch</b> <br> Codebase Indexing and Search using LLMs</h1>

SmartSearch is a advanced tool designed for efficient indexing and searching of large codebases. It combines the power of **semantic** and **keyword-based** searches with **Large Language Models (LLMs)** like **CodeBERT**, enabling developers to extract relevant code snippets quickly and effectively.


### **Key Features**
- **Local Deployment (Secure)**: Deploy locally to maintain full control over sensitive data and ensure a secure environment.

- **Model Transparency**: Provides more insight into model decisions, enhancing trust and interpretability.

- **Multi-Language Support**: Initially designed for JavaScript, Python, PHP, and Java, with the flexibility to easily extend support for additional languages.

- **FileChunking**: Processes large files into smaller, logical chunks for improved search accuracy and performance.

- **Sophisticated Scoring Method**: Ranks search results based on relevance, ensuring precise and useful outcomes.

<br>

# **After Downloading**

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

4. **Add Source Code**

   **Before using the tool, make sure to add a folder named `sourcecode` into the root (`/sourcecode/`) and add your codebase files into this folder.**

# **Using the Tool**

Once the tool is built, you can start indexing and searching your codebase with a few simple commands

# **Indexing**

Make sure to specify the directory to index (`sourcecode` - where the source code files are located).

```bash
docker-compose run --rm app python3 codebase_search.py --index sourcecode
```

# **Querying**

```bash
docker-compose run --rm app python3 codebase_search.py --search "query"
```

Examples:

- `"Where is the servername of the database?"`
- `"Where can I find the Coupon class?"`
- `"Where is the Login happening?"`
- `"Where are POST and GET requests handled?"`

# **Training**

To train your own model using your custom training data:

1. Update the `train_data.csv` file in the `train` folder with your training dataset, or specify your own file location.  
2. Adjust parameters such as learning rate, batch size, and epochs in the `train_model.py` file.

```bash
docker-compose run --rm app python3 train/train_model.py
```

# **Prerequisites**

Before using the tool, make sure your system meets the following requirements:

- **A NVIDIA GPU that supports CUDA.**
- **Windows 10/11 with WSL 2 enabled.**
- **NVIDIA Game Ready Driver version 465.89 or higher** (this is the version that supports WSL 2)

## Install Docker Desktop:

Download and install Docker Desktop for Windows.
- Ensure you enable the WSL 2 feature in Docker Desktop settings:
  - Right-click the Docker icon in the system tray and select **Settings**.
  - Go to **General** and enable the **Use the WSL 2 based engine** option.
  - In the **Resources** section, ensure that the integration with your installed WSL distributions is enabled.

## Install NVIDIA Drivers:
- Download and install the latest NVIDIA driver (Game Ready Driver) that supports CUDA from NVIDIA GeForce Experience or the NVIDIA website.

- Make sure that you have a compatible GPU

Currently, NVIDIA CUDA with WSL 2 + Docker has a bug which blocks GPU access. To fix this, use the most recent Docker Desktop version and make the following changes:

- Inside the `docker-desktop` folder, find the file `/etc/nvidia-container-runtime/config.toml` and change **`no-cgroups`** from `true` to **`false`**.

## Hyper-Training (WIP)

Hyper-Training uses 3 different parameters each and several combinations to find the best model. *(Takes a long time to run!)*

```bash
docker-compose run --rm app python3 train/hyper_train_model.py
```

This feature still needs to be fine-tuned, and is a work in progress.

# **Constraints**

This tool has been designed to handle **JavaScript**, **PHP**, and **Python** codebases. As such, several constraints are specifically tailored to these programming languages.

## General Constraints

### **Elasticsearch Integration**
- All documents are indexed in an Elasticsearch backend, which must be running and accessible.
- Proper indexing of documents ensures compatibility with semantic retrieval and question-answering pipelines.

### **Model Support**
- Uses `CodeBERT` or `RoBERTa` for embedding retrieval and question answering.
- Requires pre-trained or fine-tuned models, which should be available locally or downloaded online.

### **Tokenization and Chunking**
- Content is split into logical chunks with a limit of **512 tokens** to comply with model constraints (e.g., `CodeBERT` or `RoBERTa`).

### **Error Handling**
- Skips unreadable or inaccessible files and logs errors.

### **Scoring Mechanism**
- Combines:
  - Semantic retrieval scores.
  - Exact match scores.
  - File name relevance.
- A `synonyms` dictionary is required for query expansion.

## **File Type Constraints**

### **Ignored File Types**
The following file types are not supported and are skipped during indexing:
- `.DS_Store`, `.bin`, `.exe`, `.dll`, `.so` (system and binary files)
- `.jpg`, `.png`, `.gif` (image files)
- `.zip` (compressed archives)
- `.html` (markup files)

### **Supported Programming Languages**
The tool processes source code for:
- **JavaScript**
- **Python**
- **PHP**
- **Java**

Adjustments are to be made within `split_into_chunks` function.

## **Hardware Compatibility**
- GPU is **recommended** for optimal performance with `torch`-based models.



# **Future Improvements**

1. **Support More Languages**: Add support for C++, Ruby, Go, and other languages  
2. **Better Chunking**: Improve handling of large or mixed-language files  
3. **Faster Indexing**: Enable parallel or distributed indexing for large repositories  
4. **Additional Backends**: Support other storage systems like FAISS or Milvus  
5. **Smarter Query Expansion**: Use advanced models for more accurate search  
6. **Improved Scoring**: Enhance result ranking with neural ranking models 
7. **Custom Synonyms/Stopwords**: Extend the already used synonyms (stopword) lists. 
8. **Real-Time Updates**: Enable live indexing of new or updated files  

# CI/CD: Automated Indexing & Pre-Indexed Docker Image

## Overview

Our goal is to have every commit to the sourcecode/ directory automatically indexed by Elasticsearch, and then make the updated search index available as a Docker image. This way, anyone can run a pre-indexed Elasticsearch container without manually re-running the indexing step.

Here’s how the pipeline works, step by step:

### GitHub Actions Workflow
        The CI/CD is defined in .github/workflows/index-on-push.yml.
        It triggers on push to the sourcecode/ folder.

### Ephemeral Elasticsearch
        The workflow spins up Elasticsearch as a service (version 7.17.9 in our example).
        We wait for it to become healthy.

### Automated Indexing
        We run python codebase_search.py --index ./sourcecode to index the newly added or changed source files.
        That indexing logic uses Haystack to store documents in Elasticsearch.

### Copy Indexed Data
        After indexing, the ES container now holds a fresh index in /usr/share/elasticsearch/data.
        We copy that directory onto the GitHub runner using a docker cp command.

### Build a Pre-Indexed Image
        We have a Dockerfile.preindexed based on docker.elastic.co/elasticsearch/elasticsearch.
        In the build step, we COPY the es_data folder (which contains the newly indexed documents) into the image’s /usr/share/elasticsearch/data.
        This means the resulting Docker image has the entire Elasticsearch index baked in.

### Push to GitHub Container Registry (GHCR)
        We log in to GHCR using a GitHub secret (a Personal Access Token with write:packages scope).
        We push the newly built image to ghcr.io/<owner>/<repo>/smartsearch-es-preindexed:latest.

If a developer wants to run a code search locally, they just run docker pull on that image, start it, and Elasticsearch is up with the latest docs indexed—no manual steps.


# **Troubleshooting CUDA**

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
