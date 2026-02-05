---
icon: lucide/briefcase
---

# Use Case 2: From Decision (French) to Headnote (German)

![diagram](./images/use_case_2/diagram_light.svg#only-light)
![diagram](./images/use_case_2/diagram_dark.svg#only-dark)

Imagine you are a legal researcher, policymaker, or student working with thousands of Swiss Federal Supreme Court decisions written in German, French, or Italian (unfortunately, no Romansh). To truly understand what these cases are about, you'd need to read pages of dense legal text, interpret citations, and identify the core legal principles at play.

What if we could automatically generate concise summaries that capture the essence of each decision in any language?

The [Swiss Landmark Decisions Summarization (SLDS)](https://arxiv.org/abs/2410.13456)
 dataset ([`ipst/slds`](https://huggingface.co/datasets/ipst/slds)
) makes this possible. By the end of this section, you will see how a model trained on SLDS can take a lengthy court decision in one language and generate a concise summary in another.

### Input & Output

As mentioned above, we'll be using the [`ipst/slds`](https://huggingface.co/datasets/ipst/slds)
 dataset. Given its large size, we'll focus on a smaller subset, `fr_de`, which includes French-German pairs. You can select this subset by adding the following entry under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml#L1) section of the [`configs/ipst/slds/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml) configuration file:

```yaml
dataset_subset: fr_de
```

Each entry in the dataset contains four fields: the `decision` (the full text of the court ruling), the `decision_language` (the language in which the ruling is written), the `headnote` (the corresponding summary), and the `headnote_language` (the language of that summary).

As in the previous use case, we'll need to create `system`, `user`, and `assistant` prompts. This time, rather than revisting the JSON format, we'll focus on how to use the [`jinja`](https://jinja.palletsprojects.com/en/stable/) library to build templates that align with the entries in our dataset.

Under the [`prompts/ipst/slds`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/prompts/ipst/slds) directory, you can find the following prompts:

[`system.jinja`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/prompts/ipst/slds/system.jinja)
```jinja
You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. Your task is to generate a headnote for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is not merely a summary of the content but highlights the aspects that make the decision "leading" and important for future legislation.

When generating the headnote:
1. Focus on the core legal reasoning and key considerations that establish the decision's significance.

2. Include any relevant references to legal articles (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).

3. Use precise legal terminology and adhere to the formal and professional style typical of Swiss Federal Supreme Court headnotes.

4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand in the specified language.

Your response should consist solely of the headnote in the language specified by the user prompt.
```

[`user.jinja`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/prompts/ipst/slds/user.jinja)
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

[`assistant.jinja`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/prompts/ipst/slds/assistant.jinja)
```jinja
{{ headnote }}
```

The mapping from the raw data to the _conversational prompt–completion_ format is the same as in the previous use case. The main idea here is that [`jinja`](https://jinja.palletsprojects.com/en/stable/) templates are very flexible. <ins>You do not need to change your code for each dataset, since all dataset-specific details are handled within the templates themselves.</ins>

### Model

In [Use Case 1: From Natural Language to SQL Queries](usecase1.md), we used the lightweight [`google/gemma-3-270M-it`](google/gemma-3-270m-it) model to illustrate the fine-tuning and deployment workflow. For this second use case, we will upgrade to a more capable model, i.e., [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).

!!! warning

    As in [Use Case 1: From Natural Language to SQL Queries](#use-case-1), this is a gated model, which means you must have approved access on Hugging Face before you can download or use it. You will need to review and agree to Google's usage license on the model's page: [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).

Although [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) supports both image and text inputs (and outputs text), we will only use its text-to-text capability for this workshop. To enable this behavior, update the model class under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml#L9) section of the [`configs/ipst/slds/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml) configuration file:

```yaml
# model_class: AutoModelForCausalLM
model_class: AutoModelForImageTextToText
```

### Fine-tune

!!! tip

    If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-2`](https://huggingface.co/ALIRE-HESSO/use-case-2). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model.

If we now try to fine-tune the [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model (using the same approach as in [Use Case 1: From Natural Language to SQL Queries](usecase1.md)) with the [`configs/ipst/slds/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml) configuation, <ins>it will not work</ins>. The model is simply too large for the available GPU memory. During fine-tuning, the GPU must store the model weights, gradients, optimizer states, and temporary activations. Together, these require more memory than the GPU can provide.

Even when using the optimization trick from [Use Case 1: From Natural Language to SQL Queries](usecase1.md) with the [`configs/ipst/slds/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger.yaml) configuration, the model still does not fit in memory. Setting [`per_device_train_batch_size: 1`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger.yaml#L44) might appear to help, but this will not work either.

To overcome this limitation, we need a different fine-tuning strategy that reduces memory usage.

### Optimize: `peft`

To make fine-tuning possible on a single GPU, we will use parameter-efficient fine-tuning (PEFT) through the Hugging Face [`peft`](https://huggingface.co/docs/transformers/main/peft) library. [`peft`](https://huggingface.co/docs/transformers/main/peft) allows us to train only a small number of additional parameters while keeping the rest of the model frozen. This drastically reduces memory usage while maintaining most of the performance of full fine-tuning.

The [`configs/ipst/slds/sft_liger_peft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml) configuration builds on top of [`configs/ipst/slds/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger.yaml) but adds several important changes.

The first change is in the [`ModelConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L19) section, where we enable [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) and 4-bit quantization. LoRA adds small trainable adapters (weight matrices) to selected layers of the model, allowing it to adapt to a new task without updating all parameters.

```yaml
use_peft: true
lora_r: 32
lora_alpha: 16
lora_dropout: 0.1
load_in_4bit: true
lora_task_type: CAUSAL_LM
lora_target_modules: all-linear
```

These settings activate LoRA adapters on all linear layers ([`lora_target_modules: all-linear`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L34)) and quantize the base model to 4-bit precision ([`load_in_4bit: true`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L32)), which greatly reduces memory usage. For more details on how to choose the rank ([`lora_r`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L29)) and scaling factor alpha ([`lora_alpha`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L30)), the [LoRA without Regret](https://thinkingmachines.ai/blog/lora/) blog by [Thinking Machines](https://thinkingmachines.ai) is an excellent resource.

The second change is also in the [`ModelConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L19) section:

```yaml
attn_implementation: flash_attention_2
```

This enables [FlashAttention 2](https://arxiv.org/abs/2205.14135), an optimized attention [CUDA kernel](https://modal.com/gpu-glossary/device-software/kernel) that improves speed and reduces memory consumption during training.

The third change is in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L36) section:

```yaml
optim: adamw_torch_4bit
```

This specifies a memory-efficient optimizer that is compatible with 4-bit fine-tuning.

Still in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L36) section, we also need to increase the learning rate to help the smaller number of trainable parameters converge. Typically, this means increasing it by one order of magnitude:

```yaml
# learning_rate: 2e-5
learning_rate: 2e-4
```

Also in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L36) section, since LoRA reduces memory usage, we can increase the batch size compared to before:

```yaml
# per_device_train_batch_size: 1
per_device_train_batch_size: 4
```

!!! warning

    One could be tempted to reduce the [`max_length`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L52) parameter in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L36) section from `max_length: 8192` to a smaller value such as `max_length: 2048` to save memory, since the sequence length defines how many tokens per sample the model can process at once. However, reducing it would prevent the model from processing entire `system+user+assistant` sequences, meaning <ins>it would not fully capture</ins> the structure and meaning required for generating accurate headnotes.

### Merge

Before [deployment](usecase1.md#deploy) or [inference](usecase1.md#inference), these LoRA adapters need to be merged back into the original model. This step combines the learned task-specific adjustments from the adapters with the frozen base model weights, creating a single self-contained model that no longer depends on the [`peft`](https://huggingface.co/docs/transformers/main/peft) setup. Once merged, the resulting model behaves exactly like a standard fine-tuned model and can be used directly for [deployment](usecase1#deploy) or [inference](usecase1#inference) without loading any additional adapter files.

The merging process is controlled by the [`configs/ipst/slds/sft_liger_peft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml) configuration file. The following parameters must be added to the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L1) section to define the paths required for merging:

```yaml
peft_base_model_path: google/gemma-3-4b-it
peft_peft_model_path: ./trainer_output/google/gemma-3-4b-it-ipst/slds/checkpoint-1364
peft_output_model_path: ./trainer_output/google/gemma-3-4b-it-ipst/slds/checkpoint-1364-merged
```

where:

- [`peft_base_model_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L15) defines the original pretrained model used for fine-tuning.
- [`peft_peft_model_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L16) points to the fine-tuned model checkpoint that contains the learned LoRA adapters. The number in the folder name (for example, `checkpoint-1364`) corresponds to the training step at which the model was saved. You can select any available checkpoint depending on which version of the model you want to merge.
- [`peft_output_model_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L17) specifies where the final merged model will be saved.

Once these entries are in place, you can launch the merge process with the following command:

```bash
uv run merge.py --config configs/ipst/slds/sft_liger_peft.yaml
```

After the [`peft_output_model_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L17) script finishes, the directory defined in [`peft_output_model_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L17) will contain a fully merged and ready-to-use model that no longer depends on any external adapters.
