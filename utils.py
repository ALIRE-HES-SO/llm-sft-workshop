"""Shared helpers and dataclasses for use across training and evaluation."""
# ----------------------------------------------------------------------------------------------------
from dataclasses import dataclass
from transformers import TrainerCallback
# ----------------------------------------------------------------------------------------------------
@dataclass
class ExtraConfig:
    """Extra configuration options used by training/evaluation scripts.

    Attributes:
        mode: operation mode ("train", "evaluate", or "interact").
        model_class: model class name expected by the loader ("AutoModelForCausalLM" or "AutoModelForImageTextToText").
        dataset_name: HF dataset identifier.
        prompts_path: path to prompt templates.
        dataset_subset: optional dataset subset name.
        dataset_eval_split: split name for evaluation (use None if the dataset does not provide an evaluation split).
        dataset_test_split: split name for testing (use None if the dataset does not provide a test split).
        dataset_train_split: split name for training.
        peft_base_model_path: Base model path (the model PEFT was applied to).
        peft_peft_model_path: PEFT modules path (the checkpoint/modules produced by PEFT).
        peft_output_model_path: Merged output path (result of merging base model and PEFT modules).
        fine_tuning_data_format: format identifier used by map_dataset_format.
        evaluate_vllm_model_name_or_path: vLLM model path for evaluation.
        evaluate_vllm_sampling_params_max_tokens: max tokens for vLLM sampling.
    """
    mode: str = None
    model_class: str = None
    dataset_name: str = None
    prompts_path: str = None
    dataset_subset: str = None
    dataset_eval_split: str = None
    dataset_test_split: str = None
    dataset_train_split: str = None
    peft_base_model_path: str = None
    peft_peft_model_path: str = None
    peft_output_model_path: str = None
    fine_tuning_data_format: str = None
    evaluate_vllm_model_name_or_path: str = None
    evaluate_vllm_sampling_params_max_tokens: int = None
# ----------------------------------------------------------------------------------------------------
def map_dataset_format(
    dataset,
    user_template,
    system_template,
    assistant_template,
    fine_tuning_data_format,
):
    """Map raw dataset samples into prompt-completion pairs per the chosen format.

    Args:
        dataset: a Dataset or iterable of records to be mapped.
        user_template: a template renderer for the user message.
        system_template: a template renderer for the system message.
        assistant_template: a template renderer for the assistant response.
        fine_tuning_data_format: a string key indicating the desired output format.

    Returns:
        A mapped dataset where each sample contains "prompt" and "completion" structured
        according to the selected fine_tuning_data_format.

    Supported formats:
        - "conversational_prompt_completion": produces a list of role/content dicts for prompt
          (system + user) and a completion containing the assistant response.
    """
    match fine_tuning_data_format:
        case "conversational_prompt_completion":
            # For conversational format we build a "prompt" list (system then user)
            # and a "completion" list containing the assistant reply.
            return dataset.map(
                lambda sample: {
                    "prompt": [
                        {
                            "role": "system",
                            "content": system_template.render(**sample)
                        },
                        {
                            "role": "user",
                            "content": user_template.render(**sample)
                        }
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": assistant_template.render(**sample)
                        }
                    ]
                }
            )
        case _:
            # Explicit error for unsupported formats to aid debugging.
            raise ValueError(f"Unsupported fine_tuning_data_format: {fine_tuning_data_format}")
# ----------------------------------------------------------------------------------------------------