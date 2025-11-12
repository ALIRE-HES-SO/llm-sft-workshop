# ----------------------------------------------------------------------------------------------------
from dataclasses import dataclass
from transformers import TrainerCallback
# ----------------------------------------------------------------------------------------------------
@dataclass
class ExtraConfig:
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
    evaluate_vllm_tensor_parallel_size: int = 1
    evaluate_vllm_sampling_params_max_tokens: int = None
# ----------------------------------------------------------------------------------------------------
def map_dataset_format(
    dataset,
    user_template,
    system_template,
    assistant_template,
    fine_tuning_data_format,
):
    match fine_tuning_data_format:
        case "conversational_prompt_completion":
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
            raise ValueError(f"Unsupported fine_tuning_data_format: {fine_tuning_data_format}")