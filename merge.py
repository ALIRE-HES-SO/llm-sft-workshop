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
    
    _, _, extra_config = parser.parse_args_and_config()
    
    if extra_config.model_class not in transformers_dict:
        raise ValueError(f"Unknown model class: {extra_config.model_class}")

    model_class = transformers_dict[extra_config.model_class]
    
    logger.info("loading and saving AutoProcessor")
    AutoProcessor.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    ).save_pretrained(extra_config.peft_output_model_path)

    logger.info("loading and saving AutoTokenizer")
    AutoTokenizer.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    ).save_pretrained(extra_config.peft_output_model_path)

    logger.info("loading AutoModel")
    automodel = model_class.from_pretrained(
        extra_config.peft_base_model_path,
        device_map="cpu"
    )

    logger.info("loading PeftModel")
    peftmodel = PeftModel.from_pretrained(
        automodel,
        extra_config.peft_peft_model_path
    )
    
    logger.info("merging AutoModel and PeftModel")
    mergedmodel = peftmodel.merge_and_unload()

    logger.info("saving merged model")
    mergedmodel.save_pretrained(extra_config.peft_output_model_path)

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------