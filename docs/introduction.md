---
icon: lucide/wrench
---

# Why fine-tuning LLMs?

[Large language models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) are incredibly capable out of the box, but getting them to perform *reliably* on a specific task often requires more than just asking nicely.

Here is a **roadmap** of the techniques available you might want to try in your projects, and the reason this workshop focuses on the last one.

## Prompt engineering

Start with prompting. The first approach to try with any pre-trained model is :lucide-message-circle: [**prompt engineering**](https://en.wikipedia.org/wiki/Prompt_engineering): for example, choosing the number of examples given to the model as part of the prompt with _zero-shot_, _one-shot_ or _few-shot prompting_.

This works well for general tasks, but it has limits: you are constrained by the context window, outputs can be inconsistent across calls, and the model has no specialization for your domain-specific data or formatting requirements.

## Retrieval-augmented generation

Then consider RAG and advanced prompting. :lucide-database: [**Retrieval-augmented generation (RAG)**](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) and structured prompt templates are powerful intermediate solutions. They let you inject external knowledge at inference time, and control the output format more tightly.

However, they introduce their own challenges: retrieval errors, added latency, context length limits, and inconsistent adherence to complex output rules, especially when the task requires the model to *internalize* a style or pattern rather than just reference examples.

## Supervised fine-tuning

When you need a model to reliably reproduce domain-specific patterns, follow a strict output format, or perform a specialized task with high consistency, :lucide-sliders-horizontal: [**supervised fine-tuning (SFT)**](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) is the next step.

Fine-tuning bakes the desired behavior directly into an already-trained model's weights, by further training it on curated examples. That is exactly what this workshop teaches you to do, end to end.

!!! note

    This workshop focuses on hands-on practice rather than deep theory of fine-tuning.
    Basic familiarity with LLMs is necessary, and we will explain any advanced concepts as needed.
