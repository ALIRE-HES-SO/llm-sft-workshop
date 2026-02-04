---
icon: lucide/wrench
---

# Set up

Before diving into the fine-tuning workflows, we will first set up an Exoscale GPU instance, configure VS Code Insiders with remote tunneling to access the instance and its services, and complete the initial installation of required libraries and authentication for HuggingFace to access models and Weights & Biases to monitor training logs through its interface.

## VS Code Insiders

Throughout this workshop, we will use [VS Code Insiders](https://code.visualstudio.com/insiders/), but you are welcome to use any IDE you prefer.

We recommend [VS Code Insiders](https://code.visualstudio.com/insiders/) because it provides improved remote tunneling support, making it easier to access `localhost` services from a remote instance, including the ability to connect to locally hosted OpenAI-compatible API models.

[Download](https://code.visualstudio.com/insiders/) and install [VS Code Insiders](https://code.visualstudio.com/insiders/) before getting started.

## Exoscale instance

In this workshop, we will use `GPUA5000` instances from the `AT-VIE-2` zone. Each instance is equipped with either 1, 2, or 4 [NVIDIA RTX A5000](https://www.nvidia.com/en-us/products/workstations/rtx-a5000/) GPUs (24GB of VRAM) in the `Small`, `Medium`, and `Large` instance types, respectively. We suggest using instances with at least 100 GB of disk space.

!!! warning

    Make sure to select the `GPUA5000` instances from the `AT-VIE-2` zone!

!!! tip

    To add an [Exoscale](https://www.exoscale.com) instance have a look at the [FAQ: How can I add an Exoscale instance?](./faq).

To begin, we will use [VS Code Insiders](https://code.visualstudio.com/insiders/) to connect to a `Small` [Exoscale](https://www.exoscale.com) instance.

To set this up, open your SSH configuration file with your favorite text editor, such as `nano` or `notepad` from the terminal:

```bash
# UNIX example
nano /Users/ADD_YOUR_USERNAME_HERE]/.ssh/config
# Windows example
notepad C:\Users\YOUR_USERNAME\.ssh\config
```

!!! tip

    If the file does not exist or the command fails, you may need to generate an SSH key pair first. To do so, run the following command in your terminal:

    ```
    ssh-keygen
    # You can press Enter through all prompts to accept the default settings.
    ```

    Once the key is created, the `.ssh` directory (and the config file, if you create it) will be available.

Then add the following configuration for your [Exoscale](https://www.exoscale.com) instance:

```bash
Host exoscale
  User ubuntu
  HostName ADD_IP_HERE
```

Save and close the file. You will then be able to connect to your instance simply by running:

```bash
ssh exoscale
```

After verifying that your SSH connection works, install the [`Remote Explorer`](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer) extension in [VS Code Insiders](https://code.visualstudio.com/insiders/) to connect to your [Exoscale](https://www.exoscale.com) instance directly from the editor.

!!! tip

    If you want to keep your terminal session active even if your internet connection drops, you can use the `tmux` command. Running `tmux` creates a persistent terminal session that you can later reconnect to with: `tmux attach -t 0`. Here, `0` is the default session number, but you can create and manage multiple sessions if needed. `tmux` also allows you to split the terminal into multiple panes, which is useful for monitoring additional tools such as GPU usage. For example: `uv run nvitop`.

![tmux](./images/extra/tmux_light.png#only-light)
![tmux](./images/extra/tmux_dark.png#only-dark)

### Libraries installation and setup

We need to ensure the [Exoscale](https://www.exoscale.com) instance is correctly set up with all required tools, dependencies, and account logins.

#### GitHub CLI installation

Create a [GitHub](https://github.com) account (if you don't already have one).

Install the GitHub CLI with the following command:

```bash
sudo apt install gh
```

Once installed, authenticate your GitHub account:


```bash
gh auth login
```

Follow the on-screen instructions to complete the login process.

#### Clone and navigate to the llm-sft-workshop repository

Clone the repository:

```bash
git clone https://github.com/ALIRE-HES-SO/llm-sft-workshop
```

and navigate into its directory:

```bash
cd llm-sft-workshop
```

#### Install libraries

!!! warning

    This script will <ins>_**reboot your instance**_</ins>!

Run the installation script:

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

Then synchronize all project dependencies:

```bash
uv sync
```

#### HuggingFace CLI login

Create a [HuggingFace](https://huggingface.co) account. Then go to Profile &#8594; Acces Tokens, create a new token with `READ` permissions and copy it.

Then run the following command:

```bash
uv run hf auth login
```

When prompted, paste your token to complete the login.

!!! tip

    During login, the CLI may ask:
    ```bash
    Add token as git credential? (Y/n)
    ```
    Select: `n` and press `Enter`.

#### Weights & Biases CLI login

Create a [Weights & Biases](https://wandb.ai) account. Then go to Profile &#8594; API key, and copy it.

Then run the following command:

```bash
uv run wandb login
```

When prompted, paste your token to complete the login.

??? question "What have we just done?"

    You've installed five tools that each handle a different part of the ML workflow. Here's what each one does:

    | Tool | Purpose |
    |------|---------|
    | **GitHub CLI (`gh`)** | Clone the workshop repository to your instance |
    | **install.sh** | Install system-level dependencies (CUDA drivers, build tools) |
    | **uv** | Fast Python package manager - manages project dependencies |
    | **HuggingFace CLI (`hf`)** | Download models and datasets from the HuggingFace Hub |
    | **Weights & Biases (`wandb`)** | Track experiments, log metrics, visualize training progress |

    These tools don't communicate directly with each other. Instead, they each prepare a different piece of the puzzle:

    ```mermaid
    flowchart TD
        GitHub[GitHub]
        HF[HuggingFace Hub<br/>models, datasets]
        WB[W&B Dashboard<br/>metrics, logs]

        subgraph Instance[Your Instance]
            Install[install.sh<br/>CUDA, etc.]
            UV[uv sync<br/>Python packages]
            Training[Training Script<br/>main.py]

            Install --> UV
            UV --> Training
        end

        GitHub -->|clone| Instance
        HF <-->|read/write| Training
        Training -->|logs| WB

        style GitHub fill:#e1f5ff
        style Instance fill:#f3e5f5
        style HF fill:#e8f5e9
        style WB fill:#fff3e0
    ```

    - **gh**: Downloaded all scripts, configs, and templates to your instance
    - **install.sh**: One-time system setup. It installed CUDA drivers for GPU access and rebooted to apply changes
    - **uv sync**: Read `pyproject.toml` and installed exact versions of all Python libraries (transformers, trl, vllm, etc.)
    - **hf login**: Authenticated you with HuggingFace so training scripts can download gated models (like Gemma) and datasets automatically
    - **wandb login**: Connected your instance to your W&B dashboard so training metrics stream there in real-time

    The **training script** (`main.py`) is what ties everything together at runtime. it uses `hf` credentials to fetch data, trains using the libraries `uv` installed, and reports metrics to `wandb`.
