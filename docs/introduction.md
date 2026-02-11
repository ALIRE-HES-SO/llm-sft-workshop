---
icon: lucide/wrench
---

# Why fine-tuning LLMs?

Large language models (LLMs) are incredibly capable out of the box, but getting them to perform *reliably* on a specific task often requires more than just asking nicely.

Here is a **roadmap** of the techniques available you might want to try in your projects, and the reason this workshop focuses on the last one.

## :lucide-message-circle: Prompt engineering

Start with prompting. The first approach to try with any pre-trained model is **prompt engineering**:  _zero-shot_, _on-shot_ or _few-shot prompting_.

This works well for general tasks, but it has limits: you are constrained by the context window, outputs can be inconsistent across calls, and the model has no specialization for your domain-specific data or formatting requirements.

## :lucide-database: Retrieval-augmented generation

Then consider RAG and advanced prompting. **Retrieval-augmented generation (RAG)** and structured prompt templates are powerful intermediate solutions. They let you inject external knowledge at inference time.

However, they introduce their own challenges: retrieval errors, added latency, context length limits, and inconsistent adherence to complex output rules — especially when the task requires the model to *internalize* a style or pattern rather than just reference examples.

## :lucide-sliders-horizontal: Supervised fine-tuning

When you need a model to reliably reproduce domain-specific patterns, follow a strict output format, or perform a specialized task with high consistency, **supervised fine-tuning (SFT)** is the next step.

Fine-tuning closes the gap by training on curated examples. It bakes the desired behavior directly into the model weights. That is exactly what this workshop teaches you to do, end to end.

!!! note

    This workshop focuses on hands-on practice rather than deep theory of fine-tuning.
    Basic familiarity with LLMs is necessary, and we will explain any advanced concepts as needed.
