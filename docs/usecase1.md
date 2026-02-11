---
icon: lucide/briefcase
---

# Use Case 1: From Natural Language to SQL Queries

![diagram](./images/use_case_1/diagram_light.svg#only-light)
![diagram](./images/use_case_1/diagram_dark.svg#only-dark)

Imagine you are working with a large database full of tables, each containing dozens of columns and hundreds of rows. To explore and extract insights from this data, you usually need to write SQL queries. However, not everyone is comfortable with SQL, and this limits who can directly interact with the data.

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

!!! abstract "What you will learn"

    This first use case introduces the **complete SFT pipeline end-to-end**. You will walk through every stage:

    - Dataset preparation,
    - Training with `SFTTrainer`,
    - Optimization with `liger-kernel`,
    - Scaling to multiple GPUs with `accelerate`,
    - Deployment with `vllm`, and
    - Interaction through a chat UI implemented with `gradio`.

    The model used here (`gemma-3-270M-it`) is intentionally small with 270 million parameters, so that training stays fast and the focus remains on understanding the overall workflow rather than fighting resource constraints.

### Input & Output

[Supervised Fine-Tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) (SFT) is the process of taking a pre-trained language madel, and training further on labeled input–output pairs so it learns to produce the desired response for a given prompt. This technique allow adapting general-purpose models to specific tasks. In this workshop, we will tackle **summarization** in [Use Case 2](usecase2.md), **classification** in [Use Case 3](usecase3.md), and in this section, **translating** natural language into SQL.

To fine-tune an LLM for translating natural language into SQL, we will thus need to train it on a large dataset of example input-output pairs. This is where Hugging Face's [`datasets`](https://huggingface.co/docs/datasets/en/index) library comes in handy to manage all dataset operations, including downloading, loading, and preprocessing.

One such dataset is the [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset, which contains synthetic examples mapping natural language questions to SQL queries along with their corresponding database schemas. Each example provides a natural language question (`sql_prompt`), an associated schema and sample data (`sql_context`), and the correct SQL query (`sql`).


??? example "Dataset sample"

    ```json
    {
      "sql_prompt": "Find the total fare collected from passengers on 'Green Line' buses",
      "sql_context": "CREATE TABLE bus_routes (route_name VARCHAR(50), fare FLOAT); INSERT INTO bus_routes (route_name, fare) VALUES ('Green Line', 1.50), ('Red Line', 2.00), ('Blue Line', 1.75);",
      "sql": "SELECT SUM(fare) FROM bus_routes WHERE route_name = 'Green Line';"
    }
    ```

Now because modern instruction-tuned models are trained and inferenced on conversational data, they learn to expect prompts and responses to be structured in a chat-style format.
This means the dataset records will need to be reformatted accordingly.

For our use case, such a format would typically include three roles:

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

!!! note

    The example intentionally includes escaped newline characters (`\n`), tabs (`\t`), and SQL markdown delimiters (<code>\```sql ...```</code>). The model should learn to reproduce these elements exactly during training, as they are part of the expected output format.

The example shown here uses the _conversational prompt–completion_ format, which is one of several supported data formats for SFT, alongside _standard language modeling_, _conversational language modeling_, and _standard prompt–completion_. For more details, please refer to the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) documentation.

??? question "How is this reformatting done here?"

    The [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script, after loading the [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset from [Hugging Face](https://huggingface.co), uses the [`map_dataset_format`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/utils.py#L21) function from [`utils.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/utils.py) to map the raw dataset data into the _conversational prompt–completion_ style. It does so using [Jinja](https://jinja.palletsprojects.com/en/3.1.x/) templates located in the [`prompts/gretelai/synthetic_text_to_sql/`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/prompts/gretelai/synthetic_text_to_sql/) folder.
    
    Feel free to check the [`map_dataset_format`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/utils.py#L21) function and the corresponding templates to get an idea of how the reformatting is implemented.

### Model

In this use case, we will imagine our end goal is to deploy the fine-tuned model as part of a dashboard that users can run locally on their CPU, or even directly in the browser. We will therefore choose a relatively small model: [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it). The suffix `-it` indicates that it is _instruction-tuned_, meaning it has already been fine-tuned on chat-style, instruction-following data and can respond naturally to conversational prompts.

The model contains approximately 270 million parameters, which makes it lightweight enough for local deployment. Running the model in 16-bit precision <ins>for inference</ins> requires about 2 bytes per parameter, which translates to approximately 0.5–1 GB of (V)RAM, depending on the context length and runtime overhead from the [key–value (KV) cache](https://huggingface.co/blog/not-lain/kv-caching) used by the attention mechanism.

You will need to have approved access on Hugging Face to use it.

!!! warning "Gated model access"

    This is a _gated_ model, which means you will require access approval before use.

    * Visit its Hugging Face page at [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it)
    * Review and agree to Google's usage license.
    * Verify the page shows _"You have been granted access to this model"_.

    Once approved, you can proceed.

### Training with `trl`'s `SFTTrainer`

Now that we know what dataset and model to use, let us now look talk about the actual training.

Everything will actually be handled by the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) from the [`trl`](https://huggingface.co/docs/trl/en/index) library (short for Transformers Reinforcement Learning). This class provides a high-level interface for SFT and takes care of essentially all key steps of training: [tokenization](https://en.wikipedia.org/wiki/Large_language_model#Tokenization) (the process of converting text into numerical tokens that the model can understand), batching, checkpointing, and integration with libraries such as Weights & Biases's [`wandb`](https://wandb.ai/site/) for experiment tracking (more on this in the next section).

We already provide the code that sets up and runs the `SFTTrainer` in the [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script. It expects a configuration file (for example, [`sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml)) provided as an argument, which defines all aspects of the fine-tuning setup, including the dataset, model, and optimization parameters. This helps keep all the fine-tuning options in one place, making it easy to adjust settings without modifying the code, as we will see throughout this section and the next two use cases.

The configuration file is organized into three main sections:

- [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L1): global options such as dataset name, paths, subsets, and data formatting, used by [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py)'s custom logic.
- [`ModelConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L13): model loading options and parameter-efficient fine-tuning (PEFT) settings (which we will cover in [Use Case 2](usecase2.md)).
- [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L22): fine-tuning parameters for the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer), such as batch size, learning rate, number of epochs, logging frequency, and checkpointing strategy.

!!! tip

    Take a moment to browse through the [`configs/gretelai/synthetic_text_to_sql/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml) file to see the various options available.

### Fine-tune

We can now start the fine-tuning process.

To scale efficiently from a single GPU to multiple GPUs (as we will once we start optimising this use case), we rely on the [`accelerate`](https://huggingface.co/docs/accelerate/en/index) library. It automatically handles the distribution of training across devices, manages communication between them, and ensures that all model updates stay in sync.

It needs a configuration file that describes the device setup and the distribution strategy. For now, we will keep it simple and use a single GPU, as defined in the [`configs/accelerate_single.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_single.yaml) file, but we will see how to scale to multiple GPUs very soon.

!!! tip

    Feel free to check the contents of this file to get an idea of how the setup is defined.

We then pass it our training script [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) along with its configuration file [`configs/gretelai/synthetic_text_to_sql/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml), and `accelerate` takes care of applying our requested distribution strategy to our training script.

This boils down to the following command:

```bash
uv run accelerate launch --config_file configs/accelerate_single.yaml main.py --config configs/gretelai/synthetic_text_to_sql/sft.yaml
```

!!! info "Note"

    This command will launch training and will thus take some time. If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-1`](https://huggingface.co/ALIRE-HESSO/use-case-1). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it) model.

You can now run this command to see how the fine-tuning process starts. You should see something similar to the following:

![Fine-tune](./images/use_case_1/fine_tune_light.png#only-light)
![Fine-tune](./images/use_case_1/fine_tune_dark.png#only-dark)

??? question "A step back: what is happening?"

    The fine-tuning process is now running, and you are seeing periodic outputs of its progress.

    When running the command, the `accelerate` library launched the `main.py` script on the specified device(s) (in this case, a single GPU).
    The script loaded models from Hugging Face, as well as datasets which are transformed by Jinja into formatted prompts using the templates in `prompts/`.

    During training, `SFTTrainer` is configured to share its progress with Weights & Biases for web-based visualisation. Once complete, it saves the fine-tuned model to `training_output/`, for future use in inference as we will see.


    <figure markdown="span">
      ![Fine-tune progress](./images/use_case_1/train_mode_flow.svg)
    </figure>
    
### Monitor

Since the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) is sharing its progress with Weights & Biases (wand), you can monitor various metrics directly on the [`online dashboard`](https://wandb.ai/site/), where your run will appear with detailed metrics, logs, and charts similar to the example below:

![Fine-tune](./images/use_case_1/wandb_light.png#only-light)
![Fine-tune](./images/use_case_1/wandb_dark.png#only-dark)

You should see how the training loss decreases over time, indicating that the model is learning to generate SQL queries that better match the expected outputs.

### Optimize: `liger-kernel`

As you may have noticed, even with a relatively small model, **fine-tuning a single epoch can take quite a long time, between 100 and 110 minutes**. Can we do better? Absolutely!

Our initial setup was intentionally minimal, but various optimizations exist to improve performances. We will apply two here, starting with the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) library, which states:

> _"Liger Kernel is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%."_

No need to understand how they achieve this improvement; what matters is how easy it is to integrate it into our existing setup.

In fact, the only change needed is to add the following line to the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L22) section of our configuration file:

```yaml
use_liger_kernel: true
```

We already prepared a configuration file with this change applied right next to the original one: [`configs/gretelai/synthetic_text_to_sql/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml).

If you rerun the fine-tuning process with this new config file, you will notice **a significant reduction in memory usage** (up to 80% less), but also **a much slower execution, sometimes up to 5x slower**.

While the memory savings are valuable because they allow fine-tuning larger models (as we will see in the next use cases) without upgrading the GPU, the slowdown can be disappointing. It is beyond the scope of this workshop to explain why this happens, but you are encouraged to explore the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) library for more details.

An easy way to counteract this slowdown is to increase the batch size: each training step will process more inputs at once, thus using the GPU more efficiently. You might worry about the memory cost of increasing the batch size, but remember that the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) optimisation precisely reduced memory usage, allowing us this flexibility.

Just like enabling the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) optimisation, increasing the batch size is as simple as changing a single parameter in the configuration file's [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L22) section:

```yaml
# per_device_train_batch_size: 4
per_device_train_batch_size: 192
```

With this adjustment, the **fine-tuning time should drop to around 45 to 50 minutes, down from the original 100 to 110 minutes**. The initial slowdown caused by the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) optimisation is thus more than compensated by the increased batch size that its memory savings allowed.

### Scaling to multiple GPUs

While the combination of [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) and larger batch sizes provided a significant speedup, we can go further by tackling a more fundamental limitation: we are currently using only a single GPU.

!!! warning

    Up to this point, we have been using the `Small` instance type with a single GPU. For this section, please switch to either a `Medium` or `Large` instance, which provide 2 and 4 GPUs respectively.

    Rather than creating a new instance from scratch, you can **scale your existing instance** directly from the Exoscale console:

    1. **Stop** the current instance.
    2. Click the **three-dot menu** (top right of the instance page) and select **Scale**.
    3. Choose the desired instance type (`Medium` or `Large`).
    4. **Start** the instance again.

    This preserves your SSH keys, security groups, and all environment setup, so you can resume right where you left off.

Recall that distribution over multiple GPUs is handled by the [`accelerate`](https://huggingface.co/docs/accelerate/en/index) library, which is configured through a single YAML file.

Again, notice how minimal the required changes are to scale from a single GPU to multiple GPUs: only the following three lines of [`accelerate`](https://huggingface.co/docs/accelerate/en/index)'s configuration file [`accelerate_single.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_single.yaml) need to change.

```yaml
gpu_ids: all                # previously 0
num_processes: 4            # previously 1
distributed_type: MULTI_GPU # previously NO
```

The provided [`accelerate_multi.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml) file already includes these changes. Here is what they mean:

- `gpu_ids: all` indicates that all available GPUs should be used for training;
- `num_processes: 4` specifies that the machine can be assumed to have 4 GPUs;
- `distributed_type: MULTI_GPU` enables the multi-GPU training strategy.

!!! warning

    Make sure to adjust the `num_processes` parameter to match the number of GPUs available on your instance. For example, if you are using a `Medium` instance with 2 GPUs, set `num_processes: 2`.

To launch the fine-tuning process on multiple GPUs, simply swap the single-GPU configuration file with the multi-GPU one in the `accelerate` command. **The fine-tuning time should drop to around 10 to 15 minutes, down from the original 45 to 50 minutes**.

??? question "How does `accelerate` distribute the workload?"

    The setup in this workshop uses [`DDP`](https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html) (Distributed Data Parallel) under the hood for multi-GPU training. [`DDP`](https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html) works by creating a full copy of the model on each GPU and splitting every training batch across devices. The [`accelerate`](https://huggingface.co/docs/accelerate/en/index) library also supports more advanced distributed strategies such as [`FSDP`](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) (Fully Sharded Data Parallel) and [`Deepspeed ZeRO`](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed), which enable even larger models to be trained efficiently by sharding model parameters, gradients, and optimizer states. These methods are beyond the scope of this workshop, but you are encouraged to explore them later for large-scale fine-tuning.

### Deploy

You now have a script that can fine-tune a model in less than 15 minutes, and hopefully by now a resulting fine-tuned model ready to be deployed.

Once again, deployment is made easy by leveraging the right tools. In this workshop we will use [`vllm`](https://docs.vllm.ai/en/stable/index.html), a high-performance inference engine designed for serving LLMs efficiently and with minimal setup. It will expose the model of our choice through an OpenAI-compatible REST API, which can then be easily queried by any application.

Since [`vllm`](https://docs.vllm.ai/en/stable/index.html) is already installed as part of the project dependencies, deploying the model is as simple as running the following command:

```bash
uv run vllm serve ./trainer_output/google/gemma-3-270m-it-gretelai/synthetic_text_to_sql/checkpoint-92/ --served_model_name "local" --port 8080
```

It takes

- the path to the fine-tuned model checkpoint (in this example, we use `checkpoint-92`, but your checkpoint number may differ depending on your training setup and the number of epochs you trained for),
- the name under which the model will be served, and
- the port on which the API will be available.

Once you run this command, you can interact with your model using standard API routes, for example at: [`http://localhost:8080/v1`](http://localhost:8080/v1)

In our case, we will not interact with the API directly. Instead, we will connect it to a chat-based user interface (UI), which we will cover in the next section.

### Interact

In order to fully complete the loop from training to deployment, we created a very basic chat-based UI interface via [`gradio`](https://www.gradio.app), a Python library that allows simple creation of web-based interfaces for machine learning models.

We included the code for this interface in the same `main.py` file as the fine-tuning code. In order to use that script for interaction instead of training, you can simply change the `mode` parameter in the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L1) section of the [`configs/gretelai/synthetic_text_to_sql/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml) configuration file from `train` to `interact`.

```yaml
mode: interact # previously train
```

Making sure that the `vllm` server is running via the command in the [Deploy](#deploy) section (for example in another `tmux` or terminal tab), you can now launch the interface with:

```bash
uv run main.py --config configs/gretelai/synthetic_text_to_sql/sft_liger.yaml
```

!!! info "Note"

    Unlike the fine-tuning process, note that we do not use [`accelerate`](https://huggingface.co/docs/accelerate/en/index) here since the interface runs as a single process.

Once the server starts, open your browser and go to [`http://localhost:7860`](http://localhost:7860) to begin chatting with your locally deployed model. If you want to use the Gradio-generated link instead, go to `https://GRADIO_ID.gradio.live` and replace `GRADIO_ID` with the value shown in your terminal.

The interface should look something like this:

![diagram](./images/use_case_1/ui_light.png#only-light)
![diagram](./images/use_case_1/ui_dark.png#only-dark)

??? question "How is this working?"
    
    The `vllm` server you started loaded the fine-tuned model from the file system and exposed it via an OpenAI-compatible REST API.

    The `main.py` script is then serving a Gradio-based web UI that queries that API to obtain results of inference from the model.

    <figure markdown="span">
      ![Interact diagram](./images/use_case_1/interact_mode_flow.svg){ width="600" }
    </figure>

## What have we achieved?

This use case for a **natural-language-to-SQL** task walked through the complete SFT pipeline. Because the model is small enough to fit entirely in GPU memory, we performed **full fine-tuning** — updating every weight in the model — with only configuration files and a handful of shared scripts.

Starting from raw data, we:

- [x] loaded a dataset from Hugging Face and formatted it into chat-style [prompts using templates](#input-output),
- [x] selected a small [model](#model) suited for local and browser deployment,
- [x] [fine-tuned](#fine-tune) it using [`trl`'s `SFTTrainer`](#training-with-trls-sfttrainer), orchestrated by [`accelerate`](#fine-tune),
- [x] optimized training with [`liger-kernel`](#optimize-liger-kernel) for memory savings and larger batch sizes, then [scaled to multiple GPUs](#scaling-to-multiple-gpus),
- [x] all while [monitoring](#monitor) progress on [`wandb`](https://wandb.ai/site/).

The result is a fine-tuned model that we [deployed](#deploy) using [`vllm`](https://docs.vllm.ai/en/stable/index.html) and [interacted](#interact) with via a chat UI built with [`gradio`](https://www.gradio.app).

*[SFT]: Supervised Fine-Tuning
*[wand]: Weights & Biases