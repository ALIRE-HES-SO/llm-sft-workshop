---
icon: lucide/briefcase
---

# Use case 3: From Question to Answer

![diagram](./images/use_case_3/diagram_light.svg#only-light)
![diagram](./images/use_case_3/diagram_dark.svg#only-dark)

Imagine a medical education assistant that helps learners address complex clinical questions by analyzing scenarios, identifying key diagnostic elements, and linking them to relevant physiological or pathological mechanisms. In this use case, we leverage the [`MedQA-CoT`](https://huggingface.co/datasets/dmis-lab/meerkat-instructions/viewer/MedQA-CoT) subset of the [`dmis-lab/meerkat-instructions`](https://huggingface.co/datasets/dmis-lab/meerkat-instructions) dataset, which consists of question–answer pairs from medical board-style exams, where each item presents a clinical vignette and multiple possible answers. The `CoT` ([Chain of Thought](https://arxiv.org/abs/2201.11903)) component of the dataset provides detailed, step-by-step explanations that illustrate how each conclusion is reached. The model supports users by structuring information, emphasizing important clues, and presenting these intermediate steps clearly, making it a valuable resource for study and exam preparation.

!!! tip

    If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-3`](https://huggingface.co/ALIRE-HESSO/use-case-3). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model.


### Evaluate: bias-variance trade-off

At this point, we have covered all essential steps of SFT: [dataset preparation](usecase1.md#input-output), [optimization](usecase1.md#optimize-liger-kernel), [scaling](usecase1.md#scaling-to-multiple-gpus), and [deployment](usecase1.md#deploy). The only remaining piece is <ins>evaluation</ins>, which helps determine how well the model performs and when to stop training.

A common question is:

> _"How many epochs should I train my model for, and how do I evaluate its performance?"_

To explore this, we used the `Large` instance (4 GPUs) and fine-tuned for
multiple epochs. In the [`configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml) configuration, we first increased the number of epochs under the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L39) section:

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

You can monitor the results directly in your [`wandb`](https://wandb.ai/site/) dashboard, by creating a graph as the one below:

![Bias vs. Variance](./images/use_case_3/bias_variance_light.png#only-light)
![Bias vs. Variance](./images/use_case_3/bias_variance_dark.png#only-dark)

In this graph, the solid line represents the training loss, and the dashed line represents the validation loss. This illustrates the classic [bias–variance trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) observed in most fine-tuning setups:
- The training loss continues to decrease steadily across epochs.
- The validation loss decreases up to a certain point (around epoch 3 in this example) and then starts increasing again.
- This turning point indicates the onset of overfitting, meaning the model begins to memorize training data instead of generalizing.

You should select the checkpoint that corresponds to the lowest validation loss (for example, epoch 3, `checkpoint-393` in our case) as your final model.

### Evaluate: generalization on the test set

To evaluate the model on the test set, we provided an evaluation mode in the [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script. This mode leverages [`vllm`](https://docs.vllm.ai/en/stable/index.html) within Python itself, using the same high-performance inference engine that powers model [deployment](usecase1.md#deploy), but integrated directly in the evaluation workflow. It automatically searches for the specific pattern `the answer is (LETTER)` in both the reference and predicted answers using a regular expression, and then computes accuracy with the `exact_match` metric from the Hugging Face [`evaluate`](https://github.com/huggingface/evaluate) library.

You can enable evaluation by changing the following entry under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1) section of the [`configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml) configuration file:

```yaml
# mode: train
mode: evaluate
```

This mode also requires two additional parameters in the same [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1) section:

```yaml
evaluate_vllm_model_name_or_path: ./trainer_output/google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-395-merged
evaluate_vllm_sampling_params_max_tokens: 8192
```

where:
- [`evaluate_vllm_model_name_or_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L19) specifies the path to the merged model checkpoint that achieved the best validation performance (in this example, `checkpoint-395-merged`).
- [`evaluate_vllm_sampling_params_max_tokens`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L20) defines the maximum number of tokens the model can generate during evaluation. This value should reflect the expected verbosity of your model. For example, `CoT` models tend to produce longer outputs and may require higher limits than standard instruction models.

You can now launch the evaluation with:

```
uv run main.py --config configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml
```

!!! warning

    Unlike the fine-tuning process, we do not use [`accelerate`](https://huggingface.co/docs/accelerate/en/index) here since the evaluation runs as a single process.

After the evaluation completes, you can compare the accuracy of the baseline model (`google/gemma-3-4b-it`) against the fine-tuned model (`google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-393-merged`), as shown below:

| Model           | Accuracy (%) |
|-----------------|--------------|
| Baseline model  | 51.29        |
| Fine-tuned model| 55.79        |

The fine-tuned model achieves an accuracy of `55.79%`, outperforming the baseline by `+4.5%`. <ins>This improvement shows that even a relatively lightweight fine-tuning setup can yield measurable performance gains when adapting an instruction-tuned LLM to a specialized domain</ins>.
