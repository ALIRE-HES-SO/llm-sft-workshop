# ----------------------------------------------------------------------------------------------------
import re
import os
import logging
import evaluate
import gradio as gr
# ----------------------------------------------------------------------------------------------------
from datasets import (
    DatasetDict,
    load_dataset,
    get_dataset_config_names
)
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    __dict__ as transformers_dict
)
from trl import (
    TrlParser,
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    get_peft_config
)
from utils import (
    ExtraConfig,
    map_dataset_format
)
from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment
from vllm import (
    LLM,
    SamplingParams
)
from pprint import pprint
from dotenv import load_dotenv
from ui.chat_manager import ChatManager
# ----------------------------------------------------------------------------------------------------
load_dotenv()
# ----------------------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
# ----------------------------------------------------------------------------------------------------

def main():
    
    parser = TrlParser(
        dataclass_types=[
            SFTConfig,
            ModelConfig,
            ExtraConfig
        ]
    )
    
    sft_config, \
    model_config, \
    extra_config = parser.parse_args_and_config()

    if extra_config.model_class not in transformers_dict:
        raise ValueError(f"Unknown model class: {extra_config.model_class}")

    model_class = transformers_dict[extra_config.model_class]

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path
    )

    if extra_config.dataset_subset:
        if extra_config.dataset_subset not in get_dataset_config_names(extra_config.dataset_name):
            raise ValueError(f"Dataset subset '{extra_config.dataset_subset}' not found in dataset '{extra_config.dataset_name}'. Available subsets: {get_dataset_config_names(extra_config.dataset_name)}")
        dataset = load_dataset(extra_config.dataset_name, extra_config.dataset_subset)
    else:
        dataset = load_dataset(extra_config.dataset_name)

    if extra_config.dataset_name == "gretelai/synthetic_text_to_sql":
        import sqlparse
        dataset = dataset.map(
            lambda sample: {
                "new_sql": f"```sql\n{sqlparse.format(sample['sql'], reindent=True, keyword_case='upper')}\n```",
                "new_sql_context": f"```sql\n{sqlparse.format(sample['sql_context'], reindent=True, keyword_case='upper')}\n```"
            },
            num_proc=sft_config.dataset_num_proc
        )

    if extra_config.dataset_test_split and extra_config.dataset_eval_split is None:
        dataset_train_test_split = dataset["train"].train_test_split(
            seed=42,
            test_size=len(dataset[extra_config.dataset_test_split])
        )
        dataset = DatasetDict(
            {
                "test": dataset[extra_config.dataset_test_split],
                "train": dataset_train_test_split["train"],
                "validation": dataset_train_test_split["test"]
            }
        )
        extra_config.dataset_eval_split = "validation"
    elif extra_config.dataset_test_split is None and extra_config.dataset_eval_split is None:
        dataset_train_val_split = dataset["train"].train_test_split(
            seed=42,
            test_size=0.1
        )
        dataset_val_test_split = dataset_train_val_split["test"].train_test_split(
            seed=42,
            test_size=0.5
        )
        dataset = DatasetDict(
            {
                "test": dataset_val_test_split["test"],
                "train": dataset_train_val_split["train"],
                "validation": dataset_val_test_split["train"]
            }
        )
        extra_config.dataset_test_split = "test"
        extra_config.dataset_eval_split = "validation"

    jinja_environment = SandboxedEnvironment(
        loader=FileSystemLoader(extra_config.prompts_path)
    )
    user_template = jinja_environment.get_template("user.jinja")
    system_template = jinja_environment.get_template("system.jinja")
    assistant_template = jinja_environment.get_template("assistant.jinja")

    dataset = map_dataset_format(
        dataset=dataset,
        user_template=user_template,
        system_template=system_template,
        assistant_template=assistant_template,
        fine_tuning_data_format=extra_config.fine_tuning_data_format
    )

    match extra_config.mode:
        case "train":
            automodel = model_class.from_pretrained(
                model_config.model_name_or_path,
                dtype=model_config.dtype,
                attn_implementation=model_config.attn_implementation,
            )
            trainer = SFTTrainer(
                args=sft_config,
                model=automodel,
                peft_config=get_peft_config(model_config),
                eval_dataset=dataset[extra_config.dataset_eval_split],
                train_dataset=dataset[extra_config.dataset_train_split],
                processing_class=processor
            )
            trainer.train(
                resume_from_checkpoint=sft_config.resume_from_checkpoint
            )
        case "evaluate":
            if extra_config.dataset_subset:
                dataset_signature = f"{extra_config.dataset_name}/{extra_config.dataset_subset}"
            else:
                dataset_signature = extra_config.dataset_name
            if dataset_signature not in [
                "dmis-lab/meerkat-instructions/MedQA-CoT"
            ]:
                raise ValueError(f"Evaluation of {dataset_signature} is not yet implemented.")
            test_dataset = dataset[extra_config.dataset_test_split] 
            match extra_config.fine_tuning_data_format:
                case "conversational_prompt_completion":
                    prompts = test_dataset.map(
                        lambda sample: {
                            "text": processor.apply_chat_template(
                                sample["prompt"],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        },
                        num_proc=sft_config.dataset_num_proc
                    )["text"]
                    references = test_dataset.map(
                        lambda sample: {
                            "reference": sample["completion"][0]["content"]
                        },
                        num_proc=sft_config.dataset_num_proc
                    )["reference"]
                case _:
                    raise ValueError(f"Unsupported fine_tuning_data_format: {fine_tuning_data_format}")
            llm = LLM(
                model=extra_config.evaluate_vllm_model_name_or_path
            )
            predictions = [
                generation.outputs[0].text for generation in llm.generate(
                    prompts,
                    SamplingParams(
                        max_tokens=extra_config.evaluate_vllm_sampling_params_max_tokens,
                        temperature=0
                    )
                )
            ]

            references = [reference.strip() for reference in references]
            predictions = [prediction.strip() for prediction in predictions]
            new_references = list()
            new_predictions = list()
            for reference, prediction in zip(references, predictions):
                reference_match = re.search(r"the answer is\s*\(([A-Z])\)", reference, re.IGNORECASE)
                if reference_match:
                    new_references.append(reference_match.group(1))
                else:
                    new_references.append("X")
                prediction_match = re.search(r"the answer is\s*\(([A-Z])\)", prediction, re.IGNORECASE)
                if prediction_match:
                    new_predictions.append(prediction_match.group(1))
                else:
                    new_predictions.append("X")
            evaluation = evaluate.combine(
                [
                    "exact_match"
                ]
            )
            results = evaluation.compute(
                references=new_references,
                predictions=new_predictions
            )
            pprint(results)
        case "interact":
            if extra_config.dataset_subset:
                dataset_signature = f"{extra_config.dataset_name}/{extra_config.dataset_subset}"
            else:
                dataset_signature = extra_config.dataset_name
            if dataset_signature not in [
                "gretelai/synthetic_text_to_sql",
                "ipst/slds/fr_de",
                "dmis-lab/meerkat-instructions/MedQA-CoT"
            ]:
                raise ValueError(f"Interaction with {dataset_signature} is not yet implemented.")
            chat_manager = ChatManager(
                api_key=os.environ["OPENAI_API_KEY"],
                api_base_url=os.environ["OPENAI_API_BASE"],
                system_prompt=dataset["train"][0]["prompt"][0]["content"]
            )
            with gr.Blocks(
                title="ALIRE",
                css_paths="./ui/theme.css"
            ) as demo:
                gr.Image(
                    "./images/logo_light.svg",
                    elem_id="logo",
                    container=False,
                    show_download_button=False,
                    show_fullscreen_button=False
                )
                chatbot = gr.Chatbot(
                    type="messages",
                    elem_id="chatbot",
                    show_label=False,
                    editable="user",
                    show_copy_button=True,
                    autoscroll=True,
                    height="60vh"
                )
                with gr.Group(
                    elem_id="chat-input-group"
                ):
                    chat_input = gr.Textbox(
                        elem_id="chat-input",
                        stop_btn=False,
                        autofocus=True,
                        submit_btn=True,
                        show_label=False,
                        placeholder="Type here...",
                        html_attributes=gr.InputHTMLAttributes(
                            autocorrect="off",
                            spellcheck=False
                        )
                    )
                _examples = list()
                match dataset_signature:
                    case "gretelai/synthetic_text_to_sql":
                        _examples.append(
                            dataset["train"].filter(
                                lambda sample: sample["id"] == 5105,
                                num_proc=sft_config.dataset_num_proc
                            )[0]["prompt"][1]["content"]
                        )
                    case "ipst/slds/fr_de":
                        _examples.append(
                            dataset["train"].filter(
                                lambda sample: sample["sample_id"] == 12,
                                num_proc=sft_config.dataset_num_proc
                            )[0]["prompt"][1]["content"]
                        )
                    case "dmis-lab/meerkat-instructions/MedQA-CoT":
                        _examples.append(
                            dataset["train"].filter(
                                lambda sample: sample["id"] == "medqa-cot_17",
                                num_proc=sft_config.dataset_num_proc
                            )[0]["prompt"][1]["content"]
                        )
                    case _:
                        raise ValueError(f"Interaction with {dataset_signature} is not yet implemented.")
                for sample in dataset["test"]:
                    if len(_examples) == 4:
                        break
                    _examples.append(sample["prompt"][1]["content"])
                examples = gr.Examples(
                    elem_id="chat-examples",
                    examples=_examples,
                    inputs=[chat_input]
                )
                chat_input.submit(
                    fn=chat_manager.submit,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot],
                    show_progress="hidden",
                    api_name=False
                ).then(
                    fn=chat_manager.chat,
                    inputs=[chatbot],
                    outputs=[chatbot],
                    api_name=False
                ).then(
                    fn=chat_manager.enable_chat_input,
                    inputs=None,
                    outputs=[chat_input],
                    show_progress="hidden",
                    api_name=False
                )
                chat_input.stop(
                    fn=chat_manager.stop_chat,
                    inputs=None,
                    outputs=None,
                    show_progress="hidden",
                    api_name=False
                ).then(
                    fn=chat_manager.enable_chat_input,
                    inputs=None,
                    outputs=[chat_input],
                    show_progress="hidden",
                    api_name=False
                )
                chatbot.edit(
                    fn=chat_manager.handle_edit,
                    inputs=[chatbot],
                    outputs=[chat_input, chatbot],
                    show_progress="hidden",
                    api_name=False
                ).then(
                    fn=chat_manager.chat,
                    inputs=[chatbot],
                    outputs=[chatbot],
                    api_name=False
                ).then(
                    fn=chat_manager.enable_chat_input,
                    inputs=None,
                    outputs=[chat_input],
                    show_progress="hidden",
                    api_name=False
                )
                chatbot.retry(
                    fn=chat_manager.handle_retry,
                    inputs=[chatbot],
                    outputs=[chat_input, chatbot],
                    show_progress="hidden",
                    api_name=False
                ).then(
                    fn=chat_manager.chat,
                    inputs=[chatbot],
                    outputs=[chatbot],
                    api_name=False
                ).then(
                    fn=chat_manager.enable_chat_input,
                    inputs=None,
                    outputs=[chat_input],
                    show_progress="hidden",
                    api_name=False
                )
                demo.launch(
                    show_api=False
                )
        case _:
            raise ValueError(f"Mode {extra_config.mode} is not supported.")

if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------------------------------