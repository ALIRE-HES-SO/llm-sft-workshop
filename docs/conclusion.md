---
icon: lucide/wrench
---

# Conclusion

Throughout this workshop, you explored the complete end-to-end process of SFT for LLMs, which included [dataset preparation](usecase1.md#input-output), [fine-tuning](usecase1.md#fine-tune), [monitoring](usecase1.md#monitor), [optimization](usecase1.md#optimize-liger-kernel), [scaling](usecase1.md#scaling-to-multiple-gpus), [deployment](usecase1.md#deploy), and [interaction](usecase1.md#interact).

You learned how to:

- [x] Prepare and structure datasets into chat-style, instruction-following formats suitable for modern LLMs.
- [x] Leverage the Hugging Face ecosystem, including [`datasets`](https://huggingface.co/docs/datasets/en/index), [`trl`](https://huggingface.co/docs/trl/en/index), [`accelerate`](https://huggingface.co/docs/accelerate/en/index), and [`peft`](https://huggingface.co/docs/transformers/main/peft).
- [x] Optimize training with [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) and [`peft`](https://huggingface.co/docs/transformers/main/peft).
- [x] Scale to multiple GPUs with [`accelerate`](https://huggingface.co/docs/accelerate/en/index).
- [x] Deploy models locally with [`vllm`](https://docs.vllm.ai/en/stable/index.html).
- [x] Interact via [`gradio`](https://www.gradio.app)-based UI.
- [x] [Evaluate](usecase3.md) model performance using training and validation losses to monitor overfitting and test generalization on unseen data.

Across three practical use cases, [Use Case 1: From Natural Language to SQL Queries](usecase1.md), [Use Case 2: From Decision (French) to Headnote (German)](usecase2.md), and [Use Case 3: From Question to Answer](usecase3.md), you saw how a single, consistent SFT workflow, can adapt seamlessly across domains.

You now have a scalable, modular, and reproducible SFT pipeline that can be applied to your own projects, whether for research, prototyping, or production deployment.

We hope this workshop has given you both the practical skills and conceptual understanding needed to continue exploring, fine-tuning, and deploying your own LLMs.

*[SFT]: Supervised Fine-Tuning
*[PEFT]: Parameter-Efficient Fine-Tuning
*[wandb]: Weights & Biases