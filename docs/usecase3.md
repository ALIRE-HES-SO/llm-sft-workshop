---
icon: lucide/tally-3
---

# Evaluation
## Use Case: From Question to Answer

![diagram](./images/use_case_3/diagram_light.svg#only-light)
![diagram](./images/use_case_3/diagram_dark.svg#only-dark)

Imagine a medical education assistant that helps learners address complex clinical questions by analyzing scenarios, identifying key diagnostic elements, and linking them to relevant physiological or pathological mechanisms.

!!! abstract "What you will learn"

    This third and final use case **reuses the same PEFT setup from [Use Case 2](usecase2.md)** but focuses on the missing piece: **evaluation**. While the previous use cases covered training and deployment, this one introduces how to determine _when_ to stop training (bias–variance trade-off) and how to measure generalization on a held-out test set using the `exact_match` metric.

    The domain shifts to medical question-answering with chain-of-thought reasoning, demonstrating that the pipeline you have built generalizes across tasks.

## Input & Output

In this use case, we leverage the [`MedQA-CoT`](https://huggingface.co/datasets/dmis-lab/meerkat-instructions/viewer/MedQA-CoT) subset of the [`dmis-lab/meerkat-instructions`](https://huggingface.co/datasets/dmis-lab/meerkat-instructions) dataset, which consists of question–answer pairs from medical board-style exams, where each item presents a clinical vignette and multiple possible answers.

??? example "Dataset sample"

    ```json
    {
      "messages": [
        {
          "role": "system",
          ""content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\"."
        },
        {
          "role": "user",
          "content": "A 35-year-old male presents to his primary care physician with complaints of seasonal allergies. He has been using intranasal vasoconstrictors several times per day for several weeks. What is a likely sequela of the chronic use of topical nasal decongestants? (A) Persistent congestion (B) Permanent loss of smell (C) Persistent nasal crusting (D) Epistaxis"
        },
        {
          "role": "assistant",
          "content": "A 35-year-old male has been using intranasal vasoconstrictors for several weeks to manage his seasonal allergies. Chronic use of topical nasal decongestants can lead to a condition known as rhinitis medicamentosa. This condition is characterized by rebound congestion due to the downregulation of adrenergic receptors in the nasal mucosa, which occurs with prolonged use of these vasoconstrictors. As a result, the nasal mucosa becomes less responsive to the medication, and more of the drug is required to achieve the same effect, leading to a cycle of worsening congestion.\n\n(A) Persistent congestion is the direct sequela of chronic use of intranasal vasoconstrictors due to the reasons mentioned above.\n\n(B) Permanent loss of smell is less likely to be directly caused by the vasoconstrictors; it can be a complication of chronic rhinosinusitis or other nasal pathology but not typically from the decongestants themselves.\n\n(C) Persistent nasal crusting can occur with irritation of the nasal mucosa, but it is not the most common sequela of chronic decongestant use compared to congestion.\n\n(D) Epistaxis, or nosebleeds, can occur due to the drying effect of the nasal decongestants and the potential damage to the nasal mucosa, but it is not a unique sequela related to the chronic use as persistent congestion is.\n\nTherefore, the answer is (A) Persistent congestion."
        }
      ]
    }
    ```

The `CoT` ([Chain of Thought](https://arxiv.org/abs/2201.11903)) component of the dataset provides detailed, step-by-step explanations that illustrate how each conclusion is reached. The model supports users by structuring information, emphasizing important clues, and presenting these intermediate steps clearly, making it a valuable resource for study and exam preparation.

### Evaluate: bias-variance trade-off

At this point, we have covered all essential steps of SFT: [dataset preparation](usecase1.md#input-output), [optimization](usecase1.md#optimize-liger-kernel), [scaling](usecase1.md#scaling-to-multiple-gpus), and [deployment](usecase1.md#deploy). The only remaining piece is <ins>evaluation</ins>, which helps determine how well the model performs and when to stop training.

A common question is:

> _"How many epochs should I train my model for, and how do I evaluate its performance?"_

!!! tip

    If you want to avoid waiting for the fine-tuning process to complete, you can directly use a fine-tuned model we've already prepared for you: [`ALIRE-HESSO/use-case-3`](https://huggingface.co/ALIRE-HESSO/use-case-3). It can be used as a drop-in replacement for the fine-tuned [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) model.

To explore this, we used the `Large` instance (4 GPUs) and fine-tuned for multiple epochs. In the [`configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml) configuration, we adjusted the following parameters under the [`SFTConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L39) section:

```yaml
# num_train_epochs: 1
num_train_epochs: 10              # train for more epochs to observe overfitting
# per_device_train_batch_size: 4
per_device_train_batch_size: 16   # larger batch size to speed up training
eval_strategy: epoch              # evaluate on the validation set after each epoch
per_device_eval_batch_size: 1
```

You can monitor the results directly in your [`wandb`](https://wandb.ai/site/) dashboard, by creating a graph like the one below:

![Bias vs. Variance](./images/use_case_3/bias_variance_light.png#only-light)
![Bias vs. Variance](./images/use_case_3/bias_variance_dark.png#only-dark)

In this graph, the solid line represents the training loss, and the dashed line represents the validation loss. This illustrates the classic [bias–variance trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) observed in most fine-tuning setups:

- The training loss continues to decrease steadily across epochs.
- The validation loss decreases up to a certain point (around epoch 3 in this example) and then starts increasing again.
- This turning point indicates the onset of overfitting, meaning the model begins to memorize training data instead of generalizing.

You should select the checkpoint that corresponds to the lowest validation loss (for example, epoch 3, `checkpoint-393` in our case) as your final model.

### Evaluate: generalization on the test set

To evaluate the model on the test set, we provided an evaluation mode in the [`main.py`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/main.py) script. This mode leverages [`vllm`](https://docs.vllm.ai/en/stable/index.html) within Python itself, using the same high-performance inference engine that powers model [deployment](usecase1.md#deploy), but integrated directly in the evaluation workflow.

For each sample, the script generates a prediction and then extracts the chosen answer letter from both the reference and the predicted output using a regular expression. It then computes accuracy with the `exact_match` metric from the Hugging Face [`evaluate`](https://github.com/huggingface/evaluate) library.

??? question "How does the evaluation extract the answer?"

    The script uses the case-insensitive regex `the answer is\s*\(([A-Z])\)` to find the pattern **"the answer is (X)"** in both the reference and the model's prediction. The captured group — a single uppercase letter — is used for comparison.

    If neither the reference nor the prediction matches the pattern, a fallback placeholder `"X"` is used, ensuring that unparseable outputs never accidentally count as correct.

    Finally, the `exact_match` metric simply checks whether the extracted letter from the prediction equals the extracted letter from the reference, and reports the overall accuracy across all test samples.

You can enable evaluation by changing the following entry under the [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1) section of the [sft_liger_peft.yaml](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml) configuration file:

```yaml
# mode: train
mode: evaluate
```

This mode also requires two additional parameters in the same [`ExtraConfig`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L1) section:

```yaml
evaluate_vllm_model_name_or_path: ./trainer_output/google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-393-merged
evaluate_vllm_sampling_params_max_tokens: 8192
```

where:

- [`evaluate_vllm_model_name_or_path`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L19) specifies the path to the merged model checkpoint that achieved the best validation performance (in this example, `checkpoint-393-merged`).
- [`evaluate_vllm_sampling_params_max_tokens`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/blob/main/configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml#L20) defines the maximum number of tokens the model can generate during evaluation. This value should reflect the expected verbosity of your model. For example, `CoT` models tend to produce longer outputs and may require higher limits than standard instruction models.

You can now launch the evaluation with:

```
uv run main.py --config configs/dmis-lab/meerkat-instructions/sft_liger_peft.yaml
```

!!! warning

    Unlike the fine-tuning process, we do not use [`accelerate`](https://huggingface.co/docs/accelerate/en/index) here since the evaluation runs as a single process.

After the evaluation completes, you can compare the accuracy of the baseline model against the fine-tuned model, as shown below:

| Model      | Name      | Accuracy (%) |
|------------|-----------------|--------------|
| Baseline   |`google/gemma-3-4b-it`| 51.29        |
| Fine-tuned |`google/gemma-3-4b-it-dmis-lab/meerkat-instructions/checkpoint-393-merged`| 55.79        |

The fine-tuned model achieves an accuracy of `55.79%`, outperforming the baseline by `+4.5%`. This improvement shows that **even a relatively lightweight fine-tuning setup can yield measurable performance gains** when adapting an instruction-tuned LLM to a specialized domain.

## What have we achieved?

This final use case brought all the pieces together by applying the full SFT pipeline to a **medical question-answering** task. Using the same lightweight PEFT setup as [Use Case 2](usecase2.md), we trained only a small fraction of the model's weights, yet still achieved a measurable improvement over the baseline.

Specifically, we:

- [x] applied everything from the previous use cases to a dataset of a specialized domain requiring precise factual answers,
- [x] introduced [evaluation as the missing piece](#evaluate-bias-variance-trade-off): monitoring the bias–variance trade-off via wandb to select the best checkpoint,
- [x] ran [test-set evaluation](#evaluate-generalization-on-the-test-set) using `vllm` and the `exact_match` metric to measure real-world generalization,
- [x] demonstrated a measurable improvement (`+4.5%`) over the baseline, showing that lightweight fine-tuning yields real gains in specialized domains.

With evaluation in place, the SFT workflow is now complete, from data preparation and training to deployment and measurement.

*[SFT]: Supervised Fine-Tuning
*[PEFT]: Parameter-Efficient Fine-Tuning
*[wandb]: Weights & Biases