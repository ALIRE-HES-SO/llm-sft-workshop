#!/bin/bash

# Update package list
sudo apt update

# Install Python development headers and build tools
sudo apt install -y python3-dev
sudo apt install -y build-essential

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure ubuntu-drivers is available
sudo apt install -y ubuntu-drivers-common

# Install recommended NVIDIA driver
sudo ubuntu-drivers autoinstall

# Install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Reboot to apply driver changes
sudo reboot