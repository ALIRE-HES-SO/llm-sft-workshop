---
icon: lucide/wrench
---

# Conclusion

Throughout this workshop, you explored the complete end-to-end process of SFT for LLMs, which included [dataset preparation](#use-case-1-input-&-output), [fine-tuning](#use-case-1-fine-tune), [monitoring](#use-case-1-monitor), [optimization](#use-case-1-optimize-liger-kernel), [scaling](#use-case-1-scaling-to-multiple-gpus), [deployment](#use-case-1-deploy), and [interaction](#use-case-1-interact).

You learned how to:
- Prepare and structure datasets into chat-style, instruction-following formats suitable for modern LLMs.
- Leverage the Hugging Face ecosystem, including [`datasets`](https://huggingface.co/docs/datasets/en/index), [`trl`](https://huggingface.co/docs/trl/en/index), [`accelerate`](https://huggingface.co/docs/accelerate/en/index), and [`peft`](https://huggingface.co/docs/transformers/main/peft).
- Optimize training with [`liger-kernel`](https://github.com/linkedin/Liger-Kernel/) and [`peft`](https://huggingface.co/docs/transformers/main/peft).
- Scale to multiple GPUs with [`accelerate`](https://huggingface.co/docs/accelerate/en/index).
- Deploy models locally with [`vllm`](https://docs.vllm.ai/en/stable/index.html).
- Interact via [`gradio`](https://www.gradio.app)-based UI.
- [Evaluate](#use-case-3) model performance using training and validation losses to monitor overfitting and test generalization on unseen data.

Across three practical use cases, [Use Case 1: From Natural Language to SQL Queries](#use-case-1), [Use Case 2: From Decision (French) to Headnote (German)](#use-case-2), and [Use Case 3: From Question to Answer](#use-case-3), you saw how a single, consistent SFT workflow, can adapt seamlessly across domains.

You now have a scalable, modular, and reproducible SFT pipeline that can be applied to your own projects, whether for research, prototyping, or production deployment.

We hope this workshop has given you both the practical skills and conceptual understanding needed to continue exploring, fine-tuning, and deploying your own LLMs.