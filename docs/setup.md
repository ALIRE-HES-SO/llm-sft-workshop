---
icon: lucide/wrench
---

# Set up

## Requirements

The following requirements are necessary prior to following this workshop:

* The :material-microsoft-visual-studio-code: **[VS Code Insiders](https://code.visualstudio.com/insiders/)** editor. It provides _improved remote tunneling support_, making it easier to access `localhost` services from a remote instance.However, you are welcome to use any IDE you prefer.
* An :simple-exoscale: **[Exoscale account](https://www.exoscale.com/)** with access to GPU cloud instances. If applicable, use the provided voucher by the organizers of this workshop.
* Personal :simple-github: **[GitHub](https://github.com)**, :simple-huggingface: **[HuggingFace](https://huggingface.co)** and :simple-weightsandbiases: **[Weights & Biases](https://wandb.ai/)** accounts.

## Introduction

Because [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) requires powerful parallel compute power, we will start by setting up a GPU cloud instance on :simple-exoscale: [Exoscale](https://www.exoscale.com/), and configure an editor to access it remotely.

We will also need to install required libraries, and authenticate to external services to access models, datasets and tools (:simple-huggingface: [HuggingFace](https://huggingface.co)) and to monitor training logs visually (:simple-weightsandbiases: [Weights & Biases](https://wandb.ai/)).

We will introduce these services in more detail below.

## Set up a GPU cloud instance with Exoscale

In this workshop, we will use `GPUA5000` instances from the `AT-VIE-2` zone. Each instance is equipped with either 1, 2, or 4 [NVIDIA RTX A5000](https://www.nvidia.com/en-us/products/workstations/rtx-a5000/) GPUs (24GB of VRAM) in the `Small`, `Medium`, and `Large` instance types, respectively.

We suggest using instances with at least **100 GB** of disk space.

!!! warning

    Make sure to select the `GPUA5000` instances from the `AT-VIE-2` zone!

Create now a `Small` Exoscale instance.

!!! note

    To add an Exoscale instance, have a look at the [FAQ: How can I add an Exoscale instance?](./faq)

### Connect to the running instance

Now that your instance is up and running, connect to it using VS Code Insiders.

Open your SSH configuration file with your favorite text editor, such as `nano` or `notepad` from the terminal:

=== ":material-microsoft-windows: Windows"

    ```bash
    notepad C:\Users\YOUR_USERNAME\.ssh\config
    ```

=== ":material-apple: macOS"

    ```bash
    nano /Users/YOUR_USERNAME/.ssh/config
    ```

=== ":material-linux: Linux"

    ```bash
    nano ~/.ssh/config
    ```

??? tip "The file does not exist or the command fails? Read this!"

    If the file does not exist or the command fails, you may need to generate an SSH key pair first. To do so, run the following command in your terminal:

    ```
    ssh-keygen
    # You can press Enter through all prompts to accept the default settings.
    ```

    Once the key is created, the `.ssh` directory (and the config file, if you create it) will be available.

Then add the following configuration for your Exoscale instance.

```bash
Host exoscale
  User ubuntu
  HostName ADD_IP_HERE
```

Save and close the file.

You will now be able to connect to your instance simply by running:

```bash
ssh exoscale
```

After verifying that your SSH connection works, you need to install the [`Remote Explorer`](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer) extension in VS Code Insiders to connect to your Exoscale instance directly from the editor.

To achieve this:

* Navigate to `Extensions` (sidebar) &#8594; search for `Remote Explorer` &#8594; `install`.
* Click on `Remote Explorer` (sidedbar) and under `SSH` you should now have `exoscale` listed as an entry.
* You can then choose either `Connect in Current Window` or `Connect in New Window` using the corresponding icons next to the `exoscale` entry.

??? tip "Hint: Keeping your session active across disconnections"

    If you want to keep your terminal session active even if your internet connection drops, you can use the `tmux` command.

    Running `tmux` creates a persistent terminal session that you can later reconnect to with:

    ```bash
    tmux attach -t 0
    ```

    Here, `0` is the default session number, but you can create and manage multiple sessions if needed. `tmux` also allows you to [split the terminal into multiple panes](https://lukaszwrobel.pl/blog/tmux-tutorial-split-terminal-windows-easily/), which is useful for monitoring additional tools such as GPU usage.

    For example in the right pane one could run `uv run nvitop` and keep track of the CPU & GPU usage.

    ![tmux](./images/extra/tmux_light.png#only-light)
    ![tmux](./images/extra/tmux_dark.png#only-dark)

### Libraries installation and setup

Now that we have access to a terminal and the file system of the Exoscale instance through VS Code Insiders, we can start setting it up with all required tools, dependencies, and account logins.

We will:

- [ ] Install the GitHub CLI to clone the repository
- [ ] Install system libraries
- [ ] Install the python libraries with uv
- [ ] Authenticate to HuggingFace
- [ ] Authenticate to Weights & Biases

Take a deep breath; here we go.

#### GitHub CLI to clone the repository

In order to easily clone the workshop repository, we will use the [GitHub CLI](https://cli.github.com/).

If you don't already have one, create a [GitHub](https://github.com) account.

Install the GitHub CLI on the instance with the following command:

```bash
sudo apt install gh
```

Once installed, authenticate your GitHub account, and follow the on-screen instructions:

```bash
gh auth login
```

You can now clone the repository, and navigate into its directory:

```bash
git clone https://github.com/ALIRE-HES-SO/llm-sft-workshop
cd llm-sft-workshop
```

#### Install system libraries

To save you some time, the repository provides an installation script that will do all necessary system-level setup on your instance, which you can run with the following command:

!!! warning "This script will <ins>_**reboot your instance**_</ins>!"

    You will have to ssh back into it, and refresh the VS Code Insiders connection once it is running again.

```bash
bash ./install.sh
```

??? question "What does install.sh do?"

    This script sets up the system-level infrastructure needed for GPU-accelerated machine learning:

    - **CUDA Toolkit & Drivers**: Installs NVIDIA CUDA 12.1, which allows Python to communicate with your GPU for training
    - **Build Tools**: Installs `gcc`, `g++`, and other compilers needed to build Python packages with native extensions
    - **System Libraries**: Adds development headers and libraries that ML packages like PyTorch depend on

    The reboot is necessary because the NVIDIA kernel modules need to be loaded fresh. After rebooting, the GPU will be accessible to your training code.

#### Sync project's dependencies

After the reboot, make sure you are back inside the project directory:

```bash
cd ~/llm-sft-workshop
```

Then synchronize all project dependencies using the [uv](https://docs.astral.sh/uv/) package manager:

```bash
uv sync
```

#### HuggingFace CLI login

:simple-huggingface: [HuggingFace](https://huggingface.co) is the go-to platform for hosting and sharing machine learning models, datasets, and tools. We will use it to download pre-trained models to fine-tune, as well as datasets to train them on.

If you don't have one already, create a [HuggingFace](https://huggingface.co) account. Then go to `Profile` &#8594; `Access Tokens`, create a new token with `READ` permissions and copy it.

Then run the following command, which will prompt you for your token to complete the login.

```bash
uv run hf auth login
```

!!! tip

    During login, the CLI may ask:
    ```bash
    Add token as git credential? (Y/n)
    ```
    Select: `n` and press `Enter`.

#### Weights & Biases CLI login

:simple-weightsandbiases: [Weights & Biases](https://wandb.ai) (W&B) is a popular tool for tracking machine learning experiments and visualizing training progress in real time, which is what we will use it for in this workshop.

If you don't have one already, create a [Weights & Biases](https://wandb.ai) account. Then go to `Profile` &#8594; `API keys`, and copy it.

Then run the following command, which will prompt you for your token to complete the login.

```bash
uv run wandb login
```

## Where we are now

Your environment is now ready to start.

- [x] You have access to a [powerful GPU instance](#exoscale-instance) on the cloud, and ways to interact with it remotely through [VS Code Insiders](https://code.visualstudio.com/insiders/) and [Remote Explorer](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer).
- [x] Your instance contains the [starter code](#github-cli-to-clone-the-repository) for this workshop, with [all dependencies installed](#install-libraries).
- [x] You have accounts on the two main external services we will use ([HuggingFace](#huggingface-cli-login) and [Weights & Biases](#weights--biases-cli-login)) and have authenticated to them from the terminal.

All is ready to start fine-tuning, which we will do in the next section in a first basic use case.