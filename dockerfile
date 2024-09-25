# # Use the official Python base image
# FROM arm64v8/python:3.10.9

# # Set the working directory in the container
# WORKDIR /app

# # Copy requirements or dependencies files
# COPY requirements.txt requirements.txt

# # Update pip and install Python dependencies
# RUN pip install --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Install JupyterLab
# RUN pip install jupyterlab

# # Expose port for JupyterLab
# EXPOSE 8888

# # Copy the rest of the code into the container
# COPY . .

# # Command to run JupyterLab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
# Use the NVIDIA L4T PyTorch base image for Jetson Nano
FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /workspace

# Install necessary development packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# # Copy requirements or dependencies files
COPY requirements.txt requirements.txt
    
# Copy requirements.txt file into the container
RUN pip3 install --upgrade pip \
    && while read package; do pip3 install --no-cache-dir "$package" || echo "Failed to install $package, skipping..."; done < requirements.txt

# Install JupyterLab
RUN pip install jupyterlab

# Expose port for JupyterLab
EXPOSE 8888

# Set the default command to run JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
