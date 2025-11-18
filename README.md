<p align="center">
  <img src="./images/logo_dark.svg#gh-dark-mode-only" alt="Logo">
  <img src="./images/logo_light.svg#gh-light-mode-only" alt="Logo">
</p>

# LLM Supervised Fine-Tuning Workshop

Welcome, and thank you for your interest in this hands-on workshop organized by <b><a href="https://www.hes-so.ch/en/homepage" target="_blank">HES-SO Valais-Wallis</a></b> as part of a <b><a href="https://www.hes-so.ch/en/swiss-ai-center" target="_blank">Swiss AI Center</a></b> cheque supported by <b><a href="https://www.exoscale.com" target="_blank">Exoscale</a></b>!

In this workshop, you'll dive into the practical aspects of fine-tuning large language models (LLMs) using open-source tools and modern cloud infrastructure.

By the end of this workshop, you will have learned how to prepare datasets for fine-tuning, train models efficiently from a single GPU to multi-GPU setups, track training progress, and deploy your own LLM for practical use and experimentation.

**Points of contact:**
- Andrei Coman ([andrei.coman@hevs.ch](mailto:andrei.coman@hevs.ch))
- Pamela Delgado ([pamela.delgado@heig-vd.ch](mailto:pamela.delgado@heig-vd.ch)) 

<h2 id="table-of-contents" style="display:inline-block">Table of Contents</h2>

- [VS Code Insiders](#vs-code-insiders)
- [Exoscale instance](#exoscale-instance)
  - [Libraries installation and setup](#libraries-installation-and-setup)
- [Use Case 1: From Natural Language to SQL Queries](#use-case-1)
  - [Input & Output](#use-case-1-input-&-output) 
  - [Model](#use-case-1-model)
  - [`trl` & `accelerate`](#use-case-1-trl-&-accelerate)
  - [Fine-tune](#use-case-1-fine-tune)
  - [Monitor](#use-case-1-monitor)
  - [Optimize: `liger-kernel`](#use-case-1-optimize-liger-kernel)
  - [Scaling to multiple GPUs](#use-case-1-scaling-to-multiple-gpus)
  - [Deploy](#use-case-1-deploy)
  - [Interact](#use-case-1-interact)
- [Use Case 2: From Decision (French) to Headnote (German)](#use-case-2)
  - [Input & Output](#use-case-2-input-&-output)
  - [Model](#use-case-2-model)
  - [Fine-tune](#use-case-2-fine-tune)
  - [Optimize: `peft`](#use-case-2-optimize-peft)
  - [Merge](#use-case-2-merge)
- [Use Case 3: From Question to Answer](#use-case-3)
  - [Evaluate: bias-variance trade-off](#use-case-3-evaluate-bias-variance-tradeoff)
  - [Evaluate: generalization on the test set](#use-case-3-evaluate-generalization-on-the-test-set)
- [Conclusions](#conclusions)
- [Frequently Asked Questions](#frequently-asked-questions)
  - [How can I resume fine-tuning?](#faq-resume-fine-tuning)

<h2 id="vs-code-insiders" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> VS Code Insiders</h2>

Throughout this workshop, we will use <a href="https://code.visualstudio.com/insiders/" target="_blank">VS Code Insiders</a>, but you are welcome to use any IDE you prefer.

We recommend <a href="https://code.visualstudio.com/insiders/" target="_blank">VS Code Insiders</a> because it provides improved remote tunneling support, making it easier to access `localhost` services from a remote instance, including the ability to connect to locally hosted OpenAI-compatible API models.

<a href="https://code.visualstudio.com/insiders/" target="_blank">Download</a> and install <a href="https://code.visualstudio.com/insiders/" target="_blank">VS Code Insiders</a> before getting started.

<h2 id="exoscale-instance" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Exoscale instance</h2>

In this workshop, we will use `GPUA5000` instances from the `AT-VIE-2` zone. Each instance is equipped with either 1, 2, or 4 <a href="https://www.nvidia.com/en-us/products/workstations/rtx-a5000/" target="_blank">NVIDIA RTX A5000</a> GPUs (24GB of VRAM) in the `Small`, `Medium`, and `Large` instance types, respectively. We suggest using instances with at least 100 GB of disk space.

To begin, we will use <a href="https://code.visualstudio.com/insiders/" target="_blank">VS Code Insiders</a> to connect to a `Small` <a href="https://www.exoscale.com" target="_blank">Exoscale</a> instance.

To set this up, open your SSH configuration file with your favorite text editor, such as `nano` from the terminal:

```bash
# macOS example
nano /Users/ADD_YOUR_USERNAME_HERE]/.ssh/config
```

Then add the following configuration for your <a href="https://www.exoscale.com" target="_blank">Exoscale</a> instance:

```bash
Host exoscale
  User ADD_USERNAME_HERE
  HostName ADD_IP_HERE
```

Save and close the file. You will then be able to connect to your instance simply by running:

```bash
ssh exoscale
```

After verifying that your SSH connection works, install the <a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer" target="_blank"><code>Remote Explorer</code></a> extension in <a href="https://code.visualstudio.com/insiders/" target="_blank">VS Code Insiders</a> to connect to your <a href="https://www.exoscale.com" target="_blank">Exoscale</a> instance directly from the editor.

>💡 **Tip**
>
> If you want to keep your terminal session active even if your internet connection drops, you can use the `tmux` command. Running `tmux` creates a persistent terminal session that you can later reconnect to with: `tmux attach -t 0`. Here, `0` is the default session number, but you can create and manage multiple sessions if needed. `tmux` also allows you to split the terminal into multiple panes, which is useful for monitoring additional tools such as GPU usage. For example: `uv run nvitop`.

<p align="center">
  <img src="./images/extra/tmux_light.png#gh-light-mode-only" alt="tmux">
  <img src="./images/extra/tmux_dark.png#gh-dark-mode-only" alt="tmux">
</p>

<h3 id="libraries-installation-and-setup" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Libraries installation and setup</h3>

We need to ensure the <a href="https://www.exoscale.com" target="_blank">Exoscale</a> instance is correctly set up with all required tools, dependencies, and account logins.

#### GitHub CLI installation

Create a <a href="https://github.com" target="_blank">GitHub</a> account (if you don't already have one).

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

> ⚠️ **Nota bene**
>
> This script will <ins>_**reboot your instance**_</ins>!

Run the installation script:

```bash
bash ./install.sh
```

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

Create a <a href="https://huggingface.co" target="_blank">HuggingFace</a> account. Then go to Profile &#8594; Acces Tokens, create a new token and copy it.

Then run the following command:

```bash
uv run hf auth login
```

When prompted, paste your token to complete the login.

#### Weights & Biases CLI login

Create a <a href="https://wandb.ai" target="_blank">Weights & Biases</a> account. Then go to Profile &#8594; API key, and copy it.

Then run the following command:

```bash
uv run wandb login
```

When prompted, paste your token to complete the login.

<h2 id="use-case-1" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Use Case 1: From Natural Language to SQL Queries</h2>

<p align="center">
  <img src="./images/use_case_1/diagram_light.svg#gh-light-mode-only" alt="Diagram">
  <img src="./images/use_case_1/diagram_dark.svg#gh-dark-mode-only" alt="Diagram">
</p>

Imagine you are working with a large database full of tables, each containing dozens of columns and hundreds of rows. To explore and extract insights from this data, you usually need to write SQL queries.
However, not everyone is comfortable with SQL, and this limits who can directly interact with the data.

What if we could bridge this gap by allowing anyone to ask questions in natural language, and have a model automatically translate those questions into valid SQL queries?

By the end of this section, you will have a working model capable of taking a sentence like:

> _"Find the total fare collected from passengers on 'Green Line' buses"_

together with a schema like:

```sql
CREATE TABLE bus_routes (
    route_name VARCHAR(50),
    fare FLOAT
);

INSERT INTO bus_routes (route_name, fare)
VALUES
    ('Green Line', 1.50),
    ('Red Line', 2.00),
    ('Blue Line', 1.75);
```

and generate the corresponding SQL query:

```sql
SELECT SUM(fare)
FROM bus_routes
WHERE route_name = 'Green Line';
```

<h3 id="use-case-1-input-&-output" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Input & Output</h3>

Supervised <a href="https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)" target="_blank">fine-tuning</a> (SFT) is the process of training a pretrained language model on labeled input–output pairs so it learns to produce the desired response for a given prompt. This technique adapts general-purpose models to specific tasks such as summarization ([Use Case 2: From Decision (French) to Headnote (German)](#use-case-2)), classification ([Use Case 3: From Question to Answer](#use-case-3)), or, in this case, translating natural language into SQL.

To fine-tune an LLM for translating natural language into SQL, we first need to define how to represent our data and how the model should process it during training/inference.

We will use the Hugging Face <a href="https://huggingface.co/docs/datasets/en/index" target="_blank"><code>datasets</code></a> library to manage all dataset operations, including downloading, loading, and preprocessing.

We will use the <a href="https://huggingface.co/datasets/gretelai/synthetic_text_to_sql" target="_blank"><code>gretelai/synthetic_text_to_sql</code></a> dataset, which contains synthetic examples mapping natural language questions to SQL queries along with their corresponding database schemas. Each example, such as the one shown below, provides a natural language question (`sql_prompt`), an associated schema and sample data (`sql_context`), and the correct SQL query (`sql`).

```json
{
  "sql_prompt": "Find the total fare collected from passengers on 'Green Line' buses",
  "sql_context": "CREATE TABLE bus_routes (route_name VARCHAR(50), fare FLOAT); INSERT INTO bus_routes (route_name, fare) VALUES ('Green Line', 1.50), ('Red Line', 2.00), ('Blue Line', 1.75);",
  "sql": "SELECT SUM(fare) FROM bus_routes WHERE route_name = 'Green Line';"
}
```

To use this dataset for SFT, we need to convert each record into a chat-style format that aligns with how modern instruction-tuned models are trained/inferenced.

This format typically includes three roles:

- `system`: provides high-level context and defines the model's behavior.
- `user`: contains the actual question and the schema.
- `assistant`: contains the expected SQL query output.

For example, a single training example would be transformed as follows:

```json
{
  "system": "You are an expert SQL query generator. Your task is to generate a correct SQL query that answers a user's prompt using the provided schema.",
  "user": "Prompt:\nFind the total fare collected from passengers on 'Green Line' buses\nSchema:\nCREATE TABLE bus_routes (\n\troute_name VARCHAR(50),\n\tfare FLOAT\n);\n\nINSERT INTO bus_routes (route_name, fare)\nVALUES\n\t('Green Line', 1.50),\n\t('Red Line', 2.00),\n\t('Blue Line', 1.75);",
  "assistant": "SELECT SUM(fare)\nFROM bus_routes\nWHERE route_name = 'Green Line';"
}
```

> ⚠️ **Nota bene**
> 
> The example includes escaped newline characters (`\n`), tabs (`\t`), and SQL markdown delimiters (<code>\```sql ...```</code>). The model should learn to reproduce these elements exactly during training, as they are part of the expected output format.

The example shown here uses the _conversational prompt–completion_ format, which is one of several supported data formats for SFT, alongside _standard language modeling_, _conversational language modeling_, and _standard prompt–completion_. For more details, please refer to the Hugging Face <a href="https://huggingface.co/docs/trl/en/sft_trainer" target="_blank"><code>SFTTrainer</code></a> documentation.

> 💡 **Tip**
>
> Feel free to check the <a href="main.py" target="_blank"><code>main.py</code></a> script, which uses the <a href="utils.py#L21" target="_blank"><code>map_dataset_format</code></a> fuction from <a href="utils.py" target="_blank"><code>utils.py</code></a> to map the raw dataset data into the _conversational prompt–completion_ style.

<h3 id="use-case-1-model" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Model</h3>

Assuming the goal is to deploy this tool as part of a dashboard that might run locally on a CPU or even directly in the browser, we will use a relatively small model. The chosen model is <a href="https://huggingface.co/google/gemma-3-270m-it" target="_blank"><code>google/gemma-3-270M-it</code></a>. The suffix `-it` indicates that it is _instruction-tuned_, meaning it has already been fine-tuned on chat-style, instruction-following data and can respond naturally to conversational prompts. 

The model contains approximately 270 million parameters, which makes it lightweight enough for local deployment. Running the model in 16-bit precision <ins>for inference</ins> requires about 2 bytes per parameter, which translates to approximately 0.5–1 GB of (V)RAM, depending on the context length and runtime overhead from the <a href="https://huggingface.co/blog/not-lain/kv-caching" target="_blank">key–value (KV) cache</a> used by the attention mechanism.

> ⚠️ **Nota bene**
> 
> This is a gated model, which means you must have approved access on Hugging Face before you can download or use it. To access Gemma, you are required to review and agree to Google's usage license on the model's page: <a href="https://huggingface.co/google/gemma-3-270m-it" target="_blank"><code>google/gemma-3-270M-it</code></a>.

<h3 id="use-case-1-trl-&-accelerate" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> <code>trl</code> & <code>accelerate</code></h3>

For the fine-tuning stage, we will use the <a href="https://huggingface.co/docs/trl/en/sft_trainer" target="_blank"><code>SFTTrainer</code></a> from the Hugging Face <a href="https://huggingface.co/docs/trl/en/index" target="_blank"><code>trl</code></a> library (short for Transformers Reinforcement Learning). This class provides a high-level interface for SFT and automates many key steps such as <a href="https://en.wikipedia.org/wiki/Large_language_model#Tokenization" target="_blank">tokenization</a> (the process of converting text into numerical tokens that the model can understand), batching, checkpointing, and integration with libraries such as <a href="https://wandb.ai/site/" target="_blank"><code>wandb</code></a> (Weights & Biases) for experiment tracking (more on this in the next section).

Our entry point for running the fine-tuning process is the <a href="main.py" target="_blank"><code>main.py</code></a> script, which expects a configuration file provided as an argument (for example, <a href="configs/gretelai/synthetic_text_to_sql/sft.yaml" target="_blank"><code>sft.yaml</code></a>). This file defines all aspects of the fine-tuning setup, including the dataset, model, and optimization parameters. It keeps all the fine-tuning options in one place, making it easy to reproduce experiments or adjust settings without modifying the code.

The configuration file is organized into three main sections:

- <a href="configs/gretelai/synthetic_text_to_sql/sft.yaml#L1" target="_blank"><code>ExtraConfig</code></a>: global options such as dataset name, paths, subsets, and data formatting.
- <a href="configs/gretelai/synthetic_text_to_sql/sft.yaml#L13" target="_blank"><code>ModelConfig</code></a>: model loading options and parameter-efficient fine-tuning (PEFT) settings (more on this later).
- <a href="configs/gretelai/synthetic_text_to_sql/sft.yaml#L22" target="_blank"><code>SFTConfig</code></a>: fine-tuning parameters for the <a href="https://huggingface.co/docs/trl/en/sft_trainer" target="_blank"><code>SFTTrainer</code></a>, such as batch size, learning rate, number of epochs, logging frequency, and checkpointing strategy.

To scale efficiently from a single GPU to multiple GPUs (more on this later), we rely on the Hugging Face <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a> library. It automatically handles the distribution of training across devices, manages communication between them, and ensures that all model updates stay in sync.

<h3 id="use-case-1-fine-tune" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Fine-tune</h3>

>💡 **Tip**
>
> If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: <a href="https://huggingface.co/ALIRE-HESSO/use-case-1" target="_blank"><code>ALIRE-HESSO/use-case-1</code></a>. It can be used as a drop-in replacement for the fine-tuned <a href="https://huggingface.co/google/gemma-3-270m-it" target="_blank"><code>google/gemma-3-270M-it</code></a> model.

You can start the SFT process using the following command:

```bash
uv run accelerate launch --config_file configs/accelerate_single.yaml main.py --config configs/gretelai/synthetic_text_to_sql/sft.yaml
```

Once the job starts, your terminal should display output similar to:

<p align="center">
  <img src="./images/use_case_1/fine_tune_light.png#gh-light-mode-only" alt="Fine-tune">
  <img src="./images/use_case_1/fine_tune_dark.png#gh-dark-mode-only" alt="Fine-tune">
</p>

<h3 id="use-case-1-monitor" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Monitor</h3>

You can also monitor your training progress directly on <a href="https://wandb.ai/site/" target="_blank"><code>wandb</code></a>, where your run will appear with detailed metrics, logs, and charts similar to the example below:

<p align="center">
  <img src="./images/use_case_1/wandb_light.png#gh-light-mode-only" alt="Fine-tune">
  <img src="./images/use_case_1/wandb_dark.png#gh-dark-mode-only" alt="Fine-tune">
</p>

<h3 id="use-case-1-optimize-liger-kernel" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Optimize: <code>liger-kernel</code></h3>

As you may have noticed, even with a relatively small model, <ins>fine-tuning a single epoch can take between 100 and 110 minutes</ins>. Can we do better? Absolutely!

One can leverage the <a href="https://github.com/linkedin/Liger-Kernel/" target="_blank"><code>liger-kernel</code></a> library, which states:

> _"Liger Kernel is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%."_

To enable it, copy <a href="configs/gretelai/synthetic_text_to_sql/sft.yaml" target="_blank"><code>configs/gretelai/synthetic_text_to_sql/sft.yaml</code></a> into a new <a href="configs/gretelai/synthetic_text_to_sql/sft_liger.yaml" target="_blank"><code>configs/gretelai/synthetic_text_to_sql/sft_liger.yaml</code></a>, and add the following line in the <a href="configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L22" target="_blank"><code>SFTConfig</code></a> section:

```yaml
use_liger_kernel: true
```

If you rerun the fine-tuning process after this change, you will <ins>notice a significant reduction in memory usage (up to 80% less), but also a much slower execution, sometimes up to 5x slower</ins>.

While the memory savings are valuable because they allow fine-tuning larger models (more on this later) without upgrading the GPU, the slowdown can be disappointing. It is beyond the scope of this workshop to explain why this happens, but you are encouraged to explore the <a href="https://github.com/linkedin/Liger-Kernel/" target="_blank"><code>liger-kernel</code></a> library for more details.

To counteract the slowdown, increase the <a href="configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L41" target="_blank"><code>per_device_train_batch_size</code></a> parameter as such:

```yaml
# per_device_train_batch_size: 4
per_device_train_batch_size: 192
```

With this adjustment, the <ins>fine-tuning time should drop to around 45 to 50 minutes, down from the original 100 to 110 minutes</ins>.

<h3 id="use-case-1-scaling-to-multiple-gpus" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Scaling to multiple GPUs</h3>

To scale your fine-tuning across multiple GPUs, compare the two configuration files: <a href="configs/accelerate_single.yaml" target="_blank"><code>accelerate_single.yaml</code></a> and <a href="configs/accelerate_multi.yaml" target="_blank"><code>accelerate_multi.yaml</code></a>:

<a href="configs/accelerate_single.yaml" target="_blank"><code>accelerate_single.yaml</code></a>

```yaml
gpu_ids: 0,
num_processes: 1
distributed_type: NO
```

<a href="configs/accelerate_multi.yaml" target="_blank"><code>accelerate_multi.yaml</code></a>

```yaml
gpu_ids: all
num_processes: 4
distributed_type: MULTI_GPU
```

The multi-GPU configuration assumes your machine has 4 GPUs (<a href="configs/accelerate_multi.yaml#L2" target="_blank"><code>num_processes: 4</code></a>). The setting <a href="configs/accelerate_multi.yaml#L3" target="_blank"><code>gpu_ids: all</code></a> tells <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a> to use all available GPUs (equivalent to specifying `gpu_ids: 0,1,2,3`).

To enable multi-GPU fine-tuning, simply swap <a href="configs/accelerate_single.yaml" target="_blank"><code>accelerate_single.yaml</code></a> with <a href="configs/accelerate_multi.yaml" target="_blank"><code>accelerate_multi.yaml</code></a> during the launch of the SFT process. <ins>The fine-tuning time should drop to around 10 to 15 minutes, down from the original 45 to 50 minutes</ins>.

> 💡 **Tip**
> 
> The setup in this workshop uses <a href="https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html" target="_blank"><code>DDP</code></a> (Distributed Data Parallel) under the hood for multi-GPU training. <a href="https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html" target="_blank"><code>DDP</code></a> works by creating a full copy of the model on each GPU and splitting every training batch across devices. The <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a> library also supports more advanced distributed strategies such as <a href="https://huggingface.co/docs/accelerate/en/usage_guides/fsdp" target="_blank"><code>FSDP</code></a> (Fully Sharded Data Parallel) and <a href="https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed" target="_blank"><code>Deepspeed ZeRO</code></a>, which enable even larger models to be trained efficiently by sharding model parameters, gradients, and optimizer states. These methods are beyond the scope of this workshop, but you are encouraged to explore them later for large-scale fine-tuning.

<h3 id="use-case-1-deploy" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Deploy</h3>

To deploy the model we will use the <a href="https://docs.vllm.ai/en/stable/index.html" target="_blank"><code>vllm</code></a> library. <a href="https://docs.vllm.ai/en/stable/index.html" target="_blank"><code>vllm</code></a> is a high-performance inference engine designed for serving LLMs efficiently.

To deploy the model you just fine-tuned, run the following command:

```bash
uv run vllm serve ./trainer_output/google/gemma-3-270m-it-gretelai/synthetic_text_to_sql/checkpoint-92/ --served_model_name "local" --port 8080
```

> ⚠️ **Nota bene**
> 
> The checkpoint number (`checkpoint-92` in this example) may differ depending on your GPU configuration and the batch size used during training.

This command launches an OpenAI-compatible API endpoint that serves your model under the name `local` on port `8080`. You can interact with it using standard API routes, for example at: <a href="http://localhost:8080/v1" target="_blank"><code>http://localhost:8080/v1</code></a>

We will not interact with the API directly. Instead, we will connect it to a chat-based user interface (UI), which we will cover in the next section.

<h3 id="use-case-1-interact" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Interact</h3>

To make interaction more interesting rather than CLI based `curl` commands, we created a very basic chat-based UI interface via the Hugging Face <a href="https://www.gradio.app" target="_blank"><code>gradio</code></a> library.

You can enable the interaction by changing the following entry under the <a href="configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L1" target="_blank"><code>ExtraConfig</code></a> section of the <a href="configs/gretelai/synthetic_text_to_sql/sft_liger.yaml" target="_blank"><code>configs/gretelai/synthetic_text_to_sql/sft_liger.yaml</code></a> configuration file:

```yaml
# mode: train
mode: interact
```

To launch the interface, run the following command:

```bash
uv run main.py --config configs/gretelai/synthetic_text_to_sql/sft_liger.yaml
```

> ⚠️ **Nota bene**
> 
> Unlike the fine-tuning process, we do not use <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a> here since the interface runs as a single process.

Once the server starts, open your browser and navigate to <a href="http://localhost:7860" target="_blank"><code>http://localhost:7860</code></a> to begin chatting with your locally deployed model.

The interface should look something like this:

<p align="center">
  <img src="./images/use_case_1/ui_light.png#gh-light-mode-only" alt="ui">
  <img src="./images/use_case_1/ui_dark.png#gh-dark-mode-only" alt="ui">
</p>

<h2 id="use-case-2" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Use Case 2: From Decision (French) to Headnote (German)</h2>

<p align="center">
  <img src="./images/use_case_2/diagram_light.svg#gh-light-mode-only" alt="Diagram">
  <img src="./images/use_case_2/diagram_dark.svg#gh-dark-mode-only" alt="Diagram">
</p>

Imagine you are a legal researcher, policymaker, or student working with thousands of Swiss Federal Supreme Court decisions written in German, French, or Italian (unfortunately, no Romansh). To truly understand what these cases are about, you'd need to read pages of dense legal text, interpret citations, and identify the core legal principles at play.

What if we could automatically generate concise summaries that capture the essence of each decision in any language?

The <a href="https://arxiv.org/abs/2410.13456" target="_blank">Swiss Landmark Decisions Summarization (SLDS)</a>
 dataset (<a href="https://huggingface.co/datasets/ipst/slds" target="_blank"><code>ipst/slds</code></a>
) makes this possible. By the end of this section, you will see how a model trained on SLDS can take a lengthy court decision in one language and generate a concise summary in another.

<h3 id="use-case-2-input-&-output" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Input & Output</h3>

As mentioned above, we'll be using the <a href="https://huggingface.co/datasets/ipst/slds" target="_blank"><code>ipst/slds</code></a>
 dataset. Given its large size, we'll focus on a smaller subset, `fr_de`, which includes French-German pairs. You can select this subset by adding the following entry under the <a href="configs/ipst/slds/sft.yaml#L1" target="_blank"><code>ExtraConfig</code></a> section of the <a href="configs/ipst/slds/sft.yaml" target="_blank"><code>configs/ipst/slds/sft.yaml</code></a> configuration file:

```yaml
dataset_subset: fr_de
```

Each entry in the dataset contains four fields: the `decision` (the full text of the court ruling), the `decision_language` (the language in which the ruling is written), the `headnote` (the corresponding summary), and the `headnote_language` (the language of that summary).

As in the previous use case, we'll need to create `system`, `user`, and `assistant` prompts. This time, rather than revisting the JSON format, we'll focus on how to use the <a href="https://jinja.palletsprojects.com/en/stable/" target="_blank"><code>jinja</code></a> library to build templates that align with the entries in our dataset.

Under the <a href="prompts/ipst/slds" target="_blank"><code>prompts/ipst/slds</code></a> directory, you can find the following prompts:

<a href="prompts/ipst/slds/system.jinja" target="_blank"><code>system.jinja</code></a>
```jinja
You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. Your task is to generate a headnote for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is not merely a summary of the content but highlights the aspects that make the decision "leading" and important for future legislation.

When generating the headnote:
1. Focus on the core legal reasoning and key considerations that establish the decision's significance.

2. Include any relevant references to legal articles (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).

3. Use precise legal terminology and adhere to the formal and professional style typical of Swiss Federal Supreme Court headnotes.

4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand in the specified language.

Your response should consist solely of the headnote in the language specified by the user prompt.
```

<a href="prompts/ipst/slds/user.jinja" target="_blank"><code>user.jinja</code></a>
```jinja
Prompt:
{% if headnote_language == 'de' %}
Generate a headnote in German for the leading decision below.
{% elif headnote_language == 'fr' %}
Generate a headnote in French for the leading decision below.
{% elif headnote_language == 'it' %}
Generate a headnote in Italian for the leading decision below.
{% else %}
Generate a headnote in {{ headnote_language }} for the leading decision below.
{% endif %}

Leading decision:
{{ decision }}
```

<a href="prompts/ipst/slds/assistant.jinja" target="_blank"><code>assistant.jinja</code></a>
```jinja
{{ headnote }}
```

The mapping from the raw data to the _conversational prompt–completion_ format is the same as in the previous use case. The main idea here is that <a href="https://jinja.palletsprojects.com/en/stable/" target="_blank"><code>jinja</code></a> templates are very flexible. <ins>You do not need to change your code for each dataset, since all dataset-specific details are handled within the templates themselves.</ins>

<h3 id="use-case-2-model" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Model</h3>

In [Use Case 1: From Natural Language to SQL Queries](#use-case-1), we used the lightweight <a href="google/gemma-3-270m-it" target="_blank"><code>google/gemma-3-270M-it</code></a> model to illustrate the fine-tuning and deployment workflow. For this second use case, we will upgrade to a more capable model, i.e., <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a>.

> ⚠️ **Nota bene**
> 
> As in [Use Case 1: From Natural Language to SQL Queries](#use-case-1), this is a gated model, which means you must have approved access on Hugging Face before you can download or use it. You will need to review and agree to Google's usage license on the model's page: <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a>.

Although <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a> supports both image and text inputs (and outputs text), we will only use its text-to-text capability for this workshop. To enable this behavior, update the model class under the <a href="configs/ipst/slds/sft.yaml#L9" target="_blank"><code>ExtraConfig</code></a> section of the <a href="configs/ipst/slds/sft.yaml" target="_blank"><code>configs/ipst/slds/sft.yaml</code></a> configuration file:

```yaml
# model_class: AutoModelForCausalLM
model_class: AutoModelForImageTextToText
```

<h3 id="use-case-2-fine-tune" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Fine-tune</h3>

>💡 **Tip**
>
> If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: <a href="https://huggingface.co/ALIRE-HESSO/use-case-2" target="_blank"><code>ALIRE-HESSO/use-case-2</code></a>. It can be used as a drop-in replacement for the fine-tuned <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a> model.

If we now try to fine-tune the <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a> model (using the same approach as in [Use Case 1: From Natural Language to SQL Queries](#use-case-1)) with the <a href="configs/ipst/slds/sft.yaml" target="_blank"><code>configs/ipst/slds/sft.yaml</code></a> configuation, <ins>it will not work</ins>. The model is simply too large for the available GPU memory. During fine-tuning, the GPU must store the model weights, gradients, optimizer states, and temporary activations. Together, these require more memory than the GPU can provide.

Even when using the optimization trick from [Use Case 1: From Natural Language to SQL Queries](#use-case-1) with the <a href="configs/ipst/slds/sft_liger.yaml" target="_blank"><code>configs/ipst/slds/sft_liger.yaml</code></a> configuration, the model still does not fit in memory. Setting <a href="configs/ipst/slds/sft_liger.yaml#L44" target="_blank"><code>per_device_train_batch_size: 1</code></a> might appear to help, but this will not work either.

To overcome this limitation, we need a different fine-tuning strategy that reduces memory usage.

<h3 id="use-case-2-optimize-peft" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Optimize: <code>peft</code></h3>

To make fine-tuning possible on a single GPU, we will use parameter-efficient fine-tuning (PEFT) through the Hugging Face <a href="https://huggingface.co/docs/transformers/main/peft" target="_blank"><code>peft</code></a> library. <a href="https://huggingface.co/docs/transformers/main/peft" target="_blank"><code>peft</code></a> allows us to train only a small number of additional parameters while keeping the rest of the model frozen. This drastically reduces memory usage while maintaining most of the performance of full fine-tuning.

The <a href="configs/ipst/slds/sft_liger_peft.yaml" target="_blank"><code>configs/ipst/slds/sft_liger_peft.yaml</code></a> configuration builds on top of <a href="configs/ipst/slds/sft_liger.yaml" target="_blank"><code>configs/ipst/slds/sft_liger.yaml</code></a> but adds several important changes.

The first change is in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L19" target="_blank"><code>ModelConfig</code></a> section, where we enable <a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA (Low-Rank Adaptation)</a> and 4-bit quantization. LoRA adds small trainable adapters (weight matrices) to selected layers of the model, allowing it to adapt to a new task without updating all parameters.

```yaml
use_peft: true
lora_r: 32
lora_alpha: 16
lora_dropout: 0.1
load_in_4bit: true
lora_task_type: CAUSAL_LM
lora_target_modules: all-linear
```

These settings activate LoRA adapters on all linear layers (<a href="configs/ipst/slds/sft_liger_peft.yaml#L34" target="_blank"><code>lora_target_modules: all-linear</code></a>) and quantize the base model to 4-bit precision (<a href="configs/ipst/slds/sft_liger_peft.yaml#L32" target="_blank"><code>load_in_4bit: true</code></a>), which greatly reduces memory usage. For more details on how to choose the rank (<a href="configs/ipst/slds/sft_liger_peft.yaml#L29" target="_blank"><code>lora_r</code></a>) and scaling factor alpha (<a href="configs/ipst/slds/sft_liger_peft.yaml#L30" target="_blank"><code>lora_alpha</code></a>), the <a href="https://thinkingmachines.ai/blog/lora/" target="_blank">LoRA without Regret</a> blog by <a href="https://thinkingmachines.ai" target="_blank">Thinking Machines</a> is an excellent resource.

The second change is also in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L19" target="_blank"><code>ModelConfig</code></a> section:

```yaml
attn_implementation: flash_attention_2
```

This enables <a href="https://arxiv.org/abs/2205.14135" target="_blank">FlashAttention 2</a>, an optimized attention <a href="https://modal.com/gpu-glossary/device-software/kernel" target="_blank">CUDA kernel</a> that improves speed and reduces memory consumption during training.

The third change is in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L36" target="_blank"><code>SFTConfig</code></a> section:

```yaml
optim: adamw_torch_4bit
```

This specifies a memory-efficient optimizer that is compatible with 4-bit fine-tuning.

Still in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L36" target="_blank"><code>SFTConfig</code></a> section, we also need to increase the learning rate to help the smaller number of trainable parameters converge. Typically, this means increasing it by one order of magnitude:

```yaml
# learning_rate: 2e-5
learning_rate: 2e-4
```

Also in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L36" target="_blank"><code>SFTConfig</code></a> section, since LoRA reduces memory usage, we can increase the batch size compared to before:

```yaml
# per_device_train_batch_size: 1
per_device_train_batch_size: 4
```

> ⚠️ **Nota bene**
> 
> One could be tempted to reduce the <a href="configs/ipst/slds/sft_liger_peft.yaml#L52" target="_blank"><code>max_length</code></a> parameter in the <a href="configs/ipst/slds/sft_liger_peft.yaml#L36" target="_blank"><code>SFTConfig</code></a> section from `max_length: 8192` to a smaller value such as `max_length: 2048` to save memory, since the sequence length defines how many tokens per sample the model can process at once. However, reducing it would prevent the model from processing entire `system+user+assistant` sequences, meaning <ins>it would not fully capture</ins> the structure and meaning required for generating accurate headnotes.

<h3 id="use-case-2-merge" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Merge</h3>

Before [deployment](#use-case-1-deploy) or [inference](#use-case-1-inference), these LoRA adapters need to be merged back into the original model. This step combines the learned task-specific adjustments from the adapters with the frozen base model weights, creating a single self-contained model that no longer depends on the <a href="https://huggingface.co/docs/transformers/main/peft" target="_blank"><code>peft</code></a> setup. Once merged, the resulting model behaves exactly like a standard fine-tuned model and can be used directly for [deployment](#use-case-1-deploy) or [inference](#use-case-1-inference) without loading any additional adapter files.

The merging process is controlled by the <a href="configs/ipst/slds/sft_liger_peft.yaml" target="_blank"><code>configs/ipst/slds/sft_liger_peft.yaml</code></a> configuration file. The following parameters must be added to the <a href="configs/ipst/slds/sft_liger_peft.yaml#L1" target="_blank"><code>ExtraConfig</code></a> section to define the paths required for merging:

```yaml
peft_base_model_path: google/gemma-3-4b-it
peft_peft_model_path: ./trainer_output/google/gemma-3-4b-it-ipst/slds/checkpoint-1364
peft_output_model_path: ./trainer_output/google/gemma-3-4b-it-ipst/slds/checkpoint-1364-merged
```

where:

- <a href="configs/ipst/slds/sft_liger_peft.yaml#L15" target="_blank"><code>peft_base_model_path</code></a> defines the original pretrained model used for fine-tuning.
- <a href="configs/ipst/slds/sft_liger_peft.yaml#L16" target="_blank"><code>peft_peft_model_path</code></a> points to the fine-tuned model checkpoint that contains the learned LoRA adapters. The number in the folder name (for example, `checkpoint-1364`) corresponds to the training step at which the model was saved. You can select any available checkpoint depending on which version of the model you want to merge.
- <a href="configs/ipst/slds/sft_liger_peft.yaml#L17" target="_blank"><code>peft_output_model_path</code></a> specifies where the final merged model will be saved.

Once these entries are in place, you can launch the merge process with the following command:

```bash
uv run merge.py --config configs/ipst/slds/sft_liger_peft.yaml
```

After the <a href="merge.py" target="_blank"><code>merge.py</code></a> script finishes, the directory defined in <a href="configs/ipst/slds/sft_liger_peft.yaml#L17" target="_blank"><code>peft_output_model_path</code></a> will contain a fully merged and ready-to-use model that no longer depends on any external adapters.

<h2 id="use-case-3" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Use case 3: From Question to Answer</h2>

<p align="center">
  <img src="./images/use_case_3/diagram_light.svg#gh-light-mode-only" alt="Diagram">
  <img src="./images/use_case_3/diagram_dark.svg#gh-dark-mode-only" alt="Diagram">
</p>

Imagine a medical education assistant that helps learners address complex clinical questions by analyzing scenarios, identifying key diagnostic elements, and linking them to relevant physiological or pathological mechanisms. In this use case, we leverage the <a href="https://huggingface.co/datasets/dmis-lab/meerkat-instructions/viewer/MedQA-CoT" target="_blank"><code>MedQA-CoT</code></a> subset of the <a href="https://huggingface.co/datasets/dmis-lab/meerkat-instructions" target="_blank"><code>dmis-lab/meerkat-instructions</code></a> dataset, which consists of question–answer pairs from medical board-style exams, where each item presents a clinical vignette and multiple possible answers. The `CoT` (<a href="https://arxiv.org/abs/2201.11903" target="_blank">Chain of Thought</a>) component of the dataset provides detailed, step-by-step explanations that illustrate how each conclusion is reached. The model supports users by structuring information, emphasizing important clues, and presenting these intermediate steps clearly, making it a valuable resource for study and exam preparation.

>💡 **Tip**
>
> If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: <a href="https://huggingface.co/ALIRE-HESSO/use-case-3" target="_blank"><code>ALIRE-HESSO/use-case-3</code></a>. It can be used as a drop-in replacement for the fine-tuned <a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank"><code>google/gemma-3-4b-it</code></a> model.

<h3 id="use-case-3-evaluate-bias-variance-tradeoff" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Evaluate: bias-variance trade-off</h3>

At this point, we have covered all essential steps of SFT: [dataset preparation](#use-case-1-input-&-output), [optimization](#use-case-1-optimize-liger-kernel), [scaling](#use-case-1-scaling-to-multiple-gpus), and [deployment](#use-case-1-deploy). The only remaining piece is <ins>evaluation</ins>, which helps determine how well the model performs and when to stop training.

A common question is:

> _"How many epochs should I train my model for, and how do I evaluate its performance?"_

To explore this, we used the `Large` instance (4 GPUs) and fine-tuned for
multiple epochs. In the <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml" target="_blank"><code>configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml</code></a> configuration, we first increased the number of epochs under the <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L39" target="_blank"><code>SFTConfig</code></a> section:

```yaml
# num_train_epochs: 1
num_train_epochs: 10
```

We also increased the batch size to speed up training:

```yaml
# per_device_train_batch_size: 4
per_device_train_batch_size: 16
```

and enabled evaluation (on the `validation` set) after each epoch by adding:

```yaml
eval_strategy: epoch
per_device_eval_batch_size: 1
```

You can monitor the results directly in your <a href="https://wandb.ai/site/" target="_blank"><code>wandb</code></a> dashboard, by creating a graph as the one below:

<p align="center">
  <img src="./images/use_case_3/bias_variance_dark.png#gh-dark-mode-only" alt="Bias vs. Variance">
  <img src="./images/use_case_3/bias_variance_light.png#gh-light-mode-only" alt="Bias vs. Variance">
</p>

In this graph, the solid line represents the training loss, and the dashed line represents the validation loss. This illustrates the classic <a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff" target="_blank">bias–variance trade-off</a> observed in most fine-tuning setups:
- The training loss continues to decrease steadily across epochs.
- The validation loss decreases up to a certain point (around epoch 3 in this example) and then starts increasing again.
- This turning point indicates the onset of overfitting, meaning the model begins to memorize training data instead of generalizing.

You should select the checkpoint that corresponds to the lowest validation loss (for example, epoch 3, `checkpoint-393` in our case) as your final model.

<h3 id="use-case-3-evaluate-generalization-on-the-test-set" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Evaluate: generalization on the test set</h3>

To evaluate the model on the test set, we provided an evaluation mode in the <a href="main.py" target="_blank"><code>main.py</code></a> script. This mode leverages <a href="https://docs.vllm.ai/en/stable/index.html" target="_blank"><code>vllm</code></a> within Python itself, using the same high-performance inference engine that powers model [deployment](#use-case-1-deploy), but integrated directly in the evaluation workflow. It automatically searches for the specific pattern `the answer is (LETTER)` in both the reference and predicted answers using a regular expression, and then computes accuracy with the `exact_match` metric from the Hugging Face <a href="https://github.com/huggingface/evaluate" target="_blank"><code>evaluate</code></a> library.

You can enable evaluation by changing the following entry under the <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1" target="_blank"><code>ExtraConfig</code></a> section of the <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml" target="_blank"><code>configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml</code></a> configuration file:

```yaml
# mode: train
mode: evaluate
```

This mode also requires two additional parameters in the same <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1" target="_blank"><code>ExtraConfig</code></a> section:

```yaml
evaluate_vllm_model_name_or_path: ./trainer_output/google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-395-merged
evaluate_vllm_sampling_params_max_tokens: 8192
```

where:
- <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L19" target="_blank"><code>evaluate_vllm_model_name_or_path</code></a> specifies the path to the merged model checkpoint that achieved the best validation performance (in this example, `checkpoint-395-merged`).
- <a href="configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L20" target="_blank"><code>evaluate_vllm_sampling_params_max_tokens</code></a> defines the maximum number of tokens the model can generate during evaluation. This value should reflect the expected verbosity of your model. For example, `CoT` models tend to produce longer outputs and may require higher limits than standard instruction models.

You can now launch the evaluation with:

```
uv run main.py --config configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml
```

> ⚠️ **Nota bene**
> 
> Unlike the fine-tuning process, we do not use <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a> here since the evaluation runs as a single process.

After the evaluation completes, you can compare the accuracy of the baseline model (`google/gemma-3-4b-it`) against the fine-tuned model (`google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-393-merged`), as shown below:

| Model           | Accuracy (%) |
|-----------------|--------------|
| Baseline model  | 51.29        |
| Fine-tuned model| 55.79        |

The fine-tuned model achieves an accuracy of `55.79%`, outperforming the baseline by `+4.5%`. <ins>This improvement shows that even a relatively lightweight fine-tuning setup can yield measurable performance gains when adapting an instruction-tuned LLM to a specialized domain</ins>.

<h2 id="conclusions" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Conclusions</h2>

Throughout this workshop, you explored the complete end-to-end process of SFT for LLMs, which included [dataset preparation](#use-case-1-input-&-output), [fine-tuning](#use-case-1-fine-tune), [monitoring](#use-case-1-monitor), [optimization](#use-case-1-optimize-liger-kernel), [scaling](#use-case-1-scaling-to-multiple-gpus), [deployment](#use-case-1-deploy), and [interaction](#use-case-1-interact).

You learned how to:
- Prepare and structure datasets into chat-style, instruction-following formats suitable for modern LLMs.
- Leverage the Hugging Face ecosystem, including <a href="https://huggingface.co/docs/datasets/en/index" target="_blank"><code>datasets</code></a>, <a href="https://huggingface.co/docs/trl/en/index" target="_blank"><code>trl</code></a>, <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a>, and <a href="https://huggingface.co/docs/transformers/main/peft" target="_blank"><code>peft</code></a>.
- Optimize training with <a href="https://github.com/linkedin/Liger-Kernel/" target="_blank"><code>liger-kernel</code></a> and <a href="https://huggingface.co/docs/transformers/main/peft" target="_blank"><code>peft</code></a>.
- Scale to multiple GPUs with <a href="https://huggingface.co/docs/accelerate/en/index" target="_blank"><code>accelerate</code></a>.
- Deploy models locally with <a href="https://docs.vllm.ai/en/stable/index.html" target="_blank"><code>vllm</code></a>.
- Interact via <a href="https://www.gradio.app" target="_blank"><code>gradio</code></a>-based UI.
- [Evaluate](#use-case-3) model performance using training and validation losses to monitor overfitting and test generalization on unseen data.

Across three practical use cases, [Use Case 1: From Natural Language to SQL Queries](#use-case-1), [Use Case 2: From Decision (French) to Headnote (German)](#use-case-2), and [Use Case 3: From Question to Answer](#use-case-3), you saw how a single, consistent SFT workflow, can adapt seamlessly across domains.

You now have a scalable, modular, and reproducible SFT pipeline that can be applied to your own projects, whether for research, prototyping, or production deployment.

We hope this workshop has given you both the practical skills and conceptual understanding needed to continue exploring, fine-tuning, and deploying your own LLMs.

<h2 id="frequently-asked-questions" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> Frequently Asked Questions</h2>

<h3 id="faq-resume-fine-tuning" style="display:inline-block"><a href="#table-of-contents">&#8593;</a> How can I resume fine-tuning?</h3>

If your training process is interrupted or you want to continue from a previously saved checkpoint, you can enable checkpoint resumption in the configuration file.

Under the `SFTConfig` section, update the following parameter:

```yaml
# resume_from_checkpoint: false
resume_from_checkpoint: true
```

When this option is set to `true`, the `SFTTrainer` automatically detects the latest available checkpoint in your output directory and resumes training from that point, preserving model weights, optimizer states, and scheduler progress.

If you run the fine-tuning script as is, <a href="https://wandb.ai/site/" target="_blank"><code>wandb</code></a> will log a new run instead of continuing the previous one. To continue logging under the same initial run, you have two options:

To keep logging under the same initial run, set the environment variable before launching the script:

```bash
WANDB_RUN_ID="YOUR RUN ID" uv run accelerate launch ...
```

Alternatively, you can edit the <a href=".env" target="_blank"><code>.env</code></a> file and add the same entry:

```bash
WANDB_RUN_ID="YOUR RUN ID"
```

You can find your <a href="https://wandb.ai/site/" target="_blank"><code>wandb</code></a> run ID by opening the corresponding run page, navigating to Overview &#8594; Run path, and copying the identifier that appears after the `/`.