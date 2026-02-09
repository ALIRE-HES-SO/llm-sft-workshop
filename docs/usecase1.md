---
icon: lucide/briefcase
---

# Use Case 1: From Natural Language to SQL Queries

![diagram](./images/use_case_1/diagram_light.svg#only-light)
![diagram](./images/use_case_1/diagram_dark.svg#only-dark)

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

### Input & Output

Supervised [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) (SFT) is the process of training a pretrained language model on labeled input–output pairs so it learns to produce the desired response for a given prompt. This technique adapts general-purpose models to specific tasks such as summarization ([Use Case 2: From Decision (French) to Headnote (German)](usecase2.md)), classification ([Use Case 3: From Question to Answer](usecase3.md)), or, in this case, translating natural language into SQL.

To fine-tune an LLM for translating natural language into SQL, we first need to define how to represent our data and how the model should process it during training/inference.

We will use the [Hugging Face](https://huggingface.co) [`datasets`](https://huggingface.co/docs/datasets/en/index) library to manage all dataset operations, including downloading, loading, and preprocessing.

We will use the [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset, which contains synthetic examples mapping natural language questions to SQL queries along with their corresponding database schemas. Each example, such as the one shown below, provides a natural language question (`sql_prompt`), an associated schema and sample data (`sql_context`), and the correct SQL query (`sql`).

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

!!! warning

    The example includes escaped newline characters (`\n`), tabs (`\t`), and SQL markdown delimiters (<code>\```sql ...```</code>). The model should learn to reproduce these elements exactly during training, as they are part of the expected output format.

The example shown here uses the _conversational prompt–completion_ format, which is one of several supported data formats for SFT, alongside _standard language modeling_, _conversational language modeling_, and _standard prompt–completion_. For more details, please refer to the [Hugging Face](https://huggingface.co) [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) documentation.

!!! tip

    Feel free to check the [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script, which uses the [`map_dataset_format`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/utils.py#L21) function from [`utils.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/utils.py) to map the raw dataset data into the _conversational prompt–completion_ style.

### Model

Assuming the goal is to deploy this tool as part of a dashboard that might run locally on a CPU or even directly in the browser, we will use a relatively small model. The chosen model is [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it). The suffix `-it` indicates that it is _instruction-tuned_, meaning it has already been fine-tuned on chat-style, instruction-following data and can respond naturally to conversational prompts.

The model contains approximately 270 million parameters, which makes it lightweight enough for local deployment. Running the model in 16-bit precision <ins>for inference</ins> requires about 2 bytes per parameter, which translates to approximately 0.5–1 GB of (V)RAM, depending on the context length and runtime overhead from the [key–value (KV) cache](https://huggingface.co/blog/not-lain/kv-caching) used by the attention mechanism.

!!! warning

    This is a gated model, which means you must have approved access on [Hugging Face](https://huggingface.co) before you can download or use it. To access Gemma, you are required to review and agree to Google's usage license on the model's page: [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it).

### `trl` & `accelerate`

For the fine-tuning stage, we will use the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) from the [Hugging Face](https://huggingface.co) [`trl`](https://huggingface.co/docs/trl/en/index) library (short for Transformers Reinforcement Learning). This class provides a high-level interface for SFT and automates many key steps such as [tokenization](https://en.wikipedia.org/wiki/Large_language_model#Tokenization) (the process of converting text into numerical tokens that the model can understand), batching, checkpointing, and integration with libraries such as [`wandb`](https://wandb.ai/site/) (Weights & Biases) for experiment tracking (more on this in the next section).

Our entry point for running the fine-tuning process is the [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script, which expects a configuration file provided as an argument (for example, [`sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml)). This file defines all aspects of the fine-tuning setup, including the dataset, model, and optimization parameters. It keeps all the fine-tuning options in one place, making it easy to reproduce experiments or adjust settings without modifying the code.

The configuration file is organized into three main sections:

- [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L1): global options such as dataset name, paths, subsets, and data formatting.
- [`ModelConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L13): model loading options and parameter-efficient fine-tuning (PEFT) settings (more on this later).
- [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml#L22): fine-tuning parameters for the [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer), such as batch size, learning rate, number of epochs, logging frequency, and checkpointing strategy.

To scale efficiently from a single GPU to multiple GPUs (more on this later), we rely on the [Hugging Face](https://huggingface.co) [`accelerate`](https://huggingface.co/docs/accelerate/en/index) library. It automatically handles the distribution of training across devices, manages communication between them, and ensures that all model updates stay in sync.

### Fine-tune

!!! tip

    If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-1`](https://huggingface.co/ALIRE-HESSO/use-case-1). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-270M-it`](https://huggingface.co/google/gemma-3-270m-it) model.

You can start the SFT process using the following command:

```bash
uv run accelerate launch --config_file configs/accelerate_single.yaml main.py --config configs/gretelai/synthetic_text_to_sql/sft.yaml
```

Once the job starts, your terminal should display output similar to:

![Fine-tune](./images/use_case_1/fine_tune_light.png#only-light)
![Fine-tune](./images/use_case_1/fine_tune_dark.png#only-dark)

??? question "What are you looking at?"

    The fine-tuning process is now running, and you are seeing periodic outputs of its progress.

    When running the command, the `accelerate` library launched the `main.py` script on the specified device(s) (in this case, a single GPU).
    The script loaded models from Hugging Face, as well as datasets which are transformed by Jinja into formatted prompts using the templates in `prompts/`.

    During training, `SFTTrainer` is configured to share its progress with Weights & Biases for web-based visualisation. Once complete, it saves the fine-tuned model to `training_output/`, for future use in inference as we will see.


    <figure markdown="span">
      ![Fine-tune progress](./images/use_case_1/mode_train_light.svg#only-light){ width="600" }
    </figure>
    
### Monitor

Since progress is shared with Weights and Biases, you can monitor various metrics directly on [`wandb`](https://wandb.ai/site/), where your run will appear with detailed metrics, logs, and charts similar to the example below:

![Fine-tune](./images/use_case_1/wandb_light.png#only-light)
![Fine-tune](./images/use_case_1/wandb_dark.png#only-dark)

### Optimize: `liger-kernel`

As you may have noticed, even with a relatively small model, <ins>fine-tuning a single epoch can take between 100 and 110 minutes</ins>. Can we do better? Absolutely!

One can leverage the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) library, which states:

> _"Liger Kernel is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%."_

To enable it, copy [`configs/gretelai/synthetic_text_to_sql/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft.yaml) into a new [`configs/gretelai/synthetic_text_to_sql/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml), and add the following line in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L22) section:

```yaml
use_liger_kernel: true
```

If you rerun the fine-tuning process after this change, you will <ins>notice a significant reduction in memory usage (up to 80% less), but also a much slower execution, sometimes up to 5x slower</ins>.

While the memory savings are valuable because they allow fine-tuning larger models (more on this later) without upgrading the GPU, the slowdown can be disappointing. It is beyond the scope of this workshop to explain why this happens, but you are encouraged to explore the [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) library for more details.

To counteract the slowdown, increase the [`per_device_train_batch_size`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L41) parameter as such:

```yaml
# per_device_train_batch_size: 4
per_device_train_batch_size: 192
```

With this adjustment, the <ins>fine-tuning time should drop to around 45 to 50 minutes, down from the original 100 to 110 minutes</ins>.

### Scaling to multiple GPUs

!!! warning

    Up to this point, we have been using the `Small` instance type with a single GPU. For this section, please switch to either a `Medium` or `Large` instance, which provide 2 and 4 GPUs respectively.

To scale your fine-tuning across multiple GPUs, compare the two configuration files: [`accelerate_single.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_single.yaml) and [`accelerate_multi.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml):

[`accelerate_single.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_single.yaml)

```yaml
gpu_ids: 0,
num_processes: 1
distributed_type: NO
```

[`accelerate_multi.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml)

```yaml
gpu_ids: all
num_processes: 4
distributed_type: MULTI_GPU
```

The multi-GPU configuration assumes your machine has 4 GPUs ([`num_processes: 4`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml#L2)). The setting [`gpu_ids: all`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml#L3) tells [`accelerate`](https://huggingface.co/docs/accelerate/en/index) to use all available GPUs (equivalent to specifying `gpu_ids: 0,1,2,3`). Make sure to configure the number of GPUs to match those available on your instance.

To enable multi-GPU fine-tuning, simply swap [`accelerate_single.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_single.yaml) with [`accelerate_multi.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/accelerate_multi.yaml) during the launch of the SFT process. <ins>The fine-tuning time should drop to around 10 to 15 minutes, down from the original 45 to 50 minutes</ins>.

!!! tip

    The setup in this workshop uses [`DDP`](https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html) (Distributed Data Parallel) under the hood for multi-GPU training. [`DDP`](https://pytorch-cn.com/tutorials/intermediate/ddp_tutorial.html) works by creating a full copy of the model on each GPU and splitting every training batch across devices. The [`accelerate`](https://huggingface.co/docs/accelerate/en/index) library also supports more advanced distributed strategies such as [`FSDP`](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) (Fully Sharded Data Parallel) and [`Deepspeed ZeRO`](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed), which enable even larger models to be trained efficiently by sharding model parameters, gradients, and optimizer states. These methods are beyond the scope of this workshop, but you are encouraged to explore them later for large-scale fine-tuning.

### Deploy

To deploy the model we will use the [`vllm`](https://docs.vllm.ai/en/stable/index.html) library. [`vllm`](https://docs.vllm.ai/en/stable/index.html) is a high-performance inference engine designed for serving LLMs efficiently.

To deploy the model you just fine-tuned, run the following command:

```bash
uv run vllm serve ./trainer_output/google/gemma-3-270m-it-gretelai/synthetic_text_to_sql/checkpoint-92/ --served_model_name "local" --port 8080
```

!!! info

    The checkpoint number (`checkpoint-92` in this example) may differ depending on your GPU configuration and the batch size used during training.

This command launches an OpenAI-compatible API endpoint that serves your model under the name `local` on port `8080`. You can interact with it using standard API routes, for example at: [`http://localhost:8080/v1`](http://localhost:8080/v1)

We will not interact with the API directly. Instead, we will connect it to a chat-based user interface (UI), which we will cover in the next section.

### Interact

!!! warning

    Before proceeding with this section, make sure that the command from the [Deploy](#deploy) section is running in the background (for example, in a separate terminal tab).

To make interaction more interesting rather than CLI based `curl` commands, we created a very basic chat-based UI interface via the [Hugging Face](https://huggingface.co) [`gradio`](https://www.gradio.app) library.

You can enable the interaction by changing the following entry under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml#L1) section of the [`configs/gretelai/synthetic_text_to_sql/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/gretelai/synthetic_text_to_sql/sft_liger.yaml) configuration file:

```yaml
# mode: train
mode: interact
```

To launch the interface, run the following command:

```bash
uv run main.py --config configs/gretelai/synthetic_text_to_sql/sft_liger.yaml
```

!!! warning

    Unlike the fine-tuning process, we do not use [`accelerate`](https://huggingface.co/docs/accelerate/en/index) here since the interface runs as a single process.

Once the server starts, open your browser and go to [`http://localhost:7860`](http://localhost:7860) to begin chatting with your locally deployed model. If you want to use the Gradio-generated link instead, go to `https://GRADIO_ID.gradio.live` and replace `GRADIO_ID` with the value shown in your terminal.

The interface should look something like this:

![diagram](./images/use_case_1/ui_light.png#only-light)
![diagram](./images/use_case_1/ui_dark.png#only-dark)

??? question "How is this working?"
    
    The `vllm` server you started loaded the fine-tuned model from the file system and exposed it via an OpenAI-compatible REST API.

    The `main.py` script is then serving a Gradio-based web UI that queries that API to obtain results of inference from the model.

    <figure markdown="span">
      ![Interact diagram](./images/use_case_1/mode_interact_light.svg#only-light){ width="600" }
    </figure>

### What have we achieved?

A lot has happened here. Levering powerful tools and libraries, only moderate amounts of code were required to train a model by

- using a dataset loaded from Hugging Face to [generate prompts from templates](#input-output),
- tweaking the config files to optimize memory usage via [`liger-kernel`](#optimize-liger-kernel),
- configuring [`accelerate`](#trl--accelerate) to scale the training [across multiple GPUs](#scaling-to-multiple-gpus),
- all while monitoring progress on [`wandb`](#monitor).

This resulted in a fine-tuned model that we were able to deploy using [`vllm`](#deploy) and interact with via a simple web UI built with [`gradio`](#interact).
