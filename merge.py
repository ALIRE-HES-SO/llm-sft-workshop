# ----------------------------------------------------------------------------------------------------
import logging
# ----------------------------------------------------------------------------------------------------
from trl import (
    TrlParser,
    SFTConfig,
    ModelConfig
)
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    __dict__ as transformers_dict
)
from peft import PeftModel
from utils import ExtraConfig
# ----------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------------------------------------------

def main():
    
    # Parse command line arguments and configuration
    parser = TrlParser(
        dataclass_types=[
            SFTConfig,
            ModelConfig,
            ExtraConfig
        ]
    )
    
    # Extract the parsed configuration objects
    _, _, extra_config = parser.parse_args_and_config()
    
    # Validate that the specified model class exists in transformers_dict
    if extra_config.model_class not in transformers_dict:
        raise ValueError(f"Unknown model class: {extra_config.model_class}")

    # Get the actual model class from transformers_dict
    model_class = transformers_dict[extra_config.model_class]
    
    # Save the AutoProcessor from the base model to the output path
    logger.info("loading and saving AutoProcessor")
    AutoProcessor.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    ).save_pretrained(extra_config.peft_output_model_path)

    # Save the AutoTokenizer from the base model to the output path
    logger.info("loading and saving AutoTokenizer")
    AutoTokenizer.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    ).save_pretrained(extra_config.peft_output_model_path)

    # Load the base AutoModel from the specified path
    logger.info("loading AutoModel")
    automodel = model_class.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    )

    # Load the PEFT model from the specified path
    logger.info("loading PeftModel")
    peftmodel = PeftModel.from_pretrained(
        automodel,
        extra_config.peft_peft_model_path
    )
    
    # Merge the PEFT model with the base model and unload the PEFT adapter
    logger.info("merging AutoModel and PeftModel")
    mergedmodel = peftmodel.merge_and_unload()

    # Save the merged model to the output path
    logger.info("saving merged model")
    mergedmodel.save_pretrained(extra_config.peft_output_model_path)

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------