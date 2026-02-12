---
icon: lucide/briefcase
---

# PEFT Optimization
## Use Case: From Decision (French) to Headnote (German)

![diagram](./images/use_case_2/diagram_light.svg#only-light)
![diagram](./images/use_case_2/diagram_dark.svg#only-dark)

Imagine you are a legal researcher, policymaker, or student working with thousands of Swiss Federal Supreme Court decisions written in German, French, or Italian (unfortunately, no Romansh). To truly understand what these cases are about, you'd need to read pages of dense legal text, interpret citations, and identify the core legal principles at play.

What if we could automatically generate concise summaries that capture the essence of each decision in any language?

The [Swiss Landmark Decisions Summarization (SLDS)](https://arxiv.org/abs/2410.13456)
 dataset ([`ipst/slds`](https://huggingface.co/`datasets/ipst/slds))
makes this possible. By the end of this section, you will see how a model trained on SLDS can take a lengthy court decision in one language and generate a concise summary in another.

!!! abstract "What you will learn"

    This second use case **builds on the pipeline from [Use Case 1](usecase1.md)** but raises the stakes: the model used (`gemma-3-4b-it`) is now roughly 15x larger and no longer fits in GPU memory for full fine-tuning. This motivates the introduction of **parameter-efficient fine-tuning (PEFT)** with LoRA, 4-bit quantization, and FlashAttention 2.

    The task also changes — from SQL generation to cross-lingual summarization — yet the overall pipeline structure (dataset → format → train → deploy) stays the same, so you can see how the workflow generalizes.

### Input & Output

As mentioned above, we'll be using the [`ipst/slds`](https://huggingface.co/datasets/ipst/slds)
 dataset. Given its large size, we'll focus on a smaller subset, `fr_de`, which includes French-German pairs. You can select this subset by adding the following entry under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml#L1) section of the [`configs/ipst/slds/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml) configuration file:

```yaml
dataset_subset: fr_de
```

Each entry in the dataset contains four fields: the `decision` (the full text of the court ruling), the `decision_language` (the language in which the ruling is written), the `headnote` (the corresponding summary), and the `headnote_language` (the language of that summary).

??? example "Dataset sample"

    ```json
    {
      "decision": "Sachverhalt ab Seite 11 A.- Aux termes de l'art. 7 de la loi genevoise du 20 mai 1950 sur les agents intermédiaires, \"l'agent intermédiaire en fonds de commerce est celui qui fait profession de s'entremettre dans la vente, l'achat, la cession, la remise ou la reprise d'un fonds de commerce, quel que soit le genre de commerce exploité\". Celui qui veut [...]",
      "decision_language": "fr",
      "headnote": "Handels und Gewerbefreiheit. Polizeiliche Beschränkungen (Art. 31 BV). Vorausetzungen, unter denen die Bewilligung zur Berufsausübung, hier zur gewerbsmässigen Vermittlung von Geschäftsübertragungen, von der Erlegung einer Kaution abhängig gemacht werden kann (Erw. 2). [...]"
      "headnote_language": "de"
    }
    ```

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

The mapping from the raw data to the _conversational prompt–completion_ format is the same as in the previous use case. The main idea here is that [`jinja`](https://jinja.palletsprojects.com/en/stable/) templates are very flexible. __You do not need to change your code for each dataset, since all dataset-specific details are handled within the templates themselves.__

### Model

In [Use Case 1: From Natural Language to SQL Queries](usecase1.md), we used the lightweight [`google/gemma-3-270M-it`](google/gemma-3-270m-it) model to illustrate the fine-tuning and deployment workflow. For this second use case, we will upgrade to a more capable model, i.e., [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) with with 4 billion parameters.

!!! warning "Gated model access"

    As in [Use Case 1](usecase1.md), this is a _gated_ model, which means you will require access approval before use.

    * Visit its Hugging Face page at [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).
    * Review and agree to Google's usage license.
    * Verify the page shows _"You have been granted access to this model"_.

    Once approved, you can proceed.

Although [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) supports both image and text inputs (and outputs text), we will only use its text-to-text capability for this workshop. To enable this behavior, update the model class under the `ExtraConfig` section of `configs/ipst/slds/sft.yaml`:

```yaml
# model_class: AutoModelForCausalLM
model_class: AutoModelForImageTextToText
```

### Fine-tune

!!! tip

    If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-2`](https://huggingface.co/ALIRE-HESSO/use-case-2). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model.

If we now try to fine-tune [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) using the same approach as in [Use Case 1](usecase1.md) with the [`configs/ipst/slds/sft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft.yaml) configuation, **it will not work**. The model is simply too large for the available GPU memory. During fine-tuning, the GPU must store the model weights, gradients, optimizer states, and temporary activations.

Even with the Liger optimization ([`configs/ipst/slds/sft_liger.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger.yaml)) and `per_device_train_batch_size: 1` setting, the model still does not fit in memory.

We need a different fine-tuning strategy that reduces memory usage.

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

The second change is also in the `ModelConfig` section:

```yaml
attn_implementation: flash_attention_2
```

This enables [FlashAttention 2](https://arxiv.org/abs/2205.14135), an optimized attention [CUDA kernel](https://modal.com/gpu-glossary/device-software/kernel) that improves speed and reduces memory consumption during training.

The third change is in the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/ipst/slds/sft_liger_peft.yaml#L36) section:

```yaml
optim: adamw_torch_4bit
```

This specifies a memory-efficient optimizer that is compatible with 4-bit fine-tuning.

Still in the `SFTConfig` section, we also adjust the learning rate and batch size. With fewer trainable parameters, a higher learning rate (typically one order of magnitude) helps convergence; and since LoRA reduces memory usage, we can afford a larger batch size:

```yaml
learning_rate: 2e-4              # increased from 2e-5 — helps fewer trainable params converge
per_device_train_batch_size: 4   # increased from 1 — LoRA frees enough memory
```

!!! warning

    One could be tempted to reduce the `max_length` parameter in the `SFTConfig` section from `max_length: 8192` to a smaller value such as `max_length: 2048` to save memory, since the sequence length defines how many tokens per sample the model can process at once. However, reducing it would prevent the model from processing entire `system+user+assistant` sequences, meaning **it would not fully capture** the structure and meaning required for generating accurate headnotes.

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

After the merge script finishes, the directory defined in `peft_output_model_path` will contain a fully merged and ready-to-use model that no longer depends on any external adapters.

## What have we achieved?

This use case pushed the boundaries of what we saw in Use Case 1. We tackled a **cross-lingual summarization** task — generating German headnotes from French court decisions.

Thanks to LoRA and 4-bit quantization, only a small fraction of the 4B-parameter model was actually trained, while the bulk of the weights stayed frozen, by:

- [x] preparing [input & output](#input-output) pairs from the [`ipst/slds`](https://huggingface.co/datasets/ipst/slds) dataset using Jinja templates,
- [x] upgrading to a larger [model](#model) ([`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)) and hitting GPU memory limits in the process,
- [x] overcoming those limits with [parameter-efficient fine-tuning (`peft`)](#optimize-peft) Using LoRA and 4-bit quantization, combined with FlashAttention 2,
- [x] [merging](#merge) the LoRA adapters back into the base model for seamless deployment.

The result is a fully self-contained fine-tuned model, ready to deploy without any adapter dependencies.
