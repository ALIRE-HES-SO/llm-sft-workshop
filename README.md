# llm-sft-workshop

Hands-on workshop organized by **[HES-SO Valais-Wallis](https://www.hes-so.ch/en/homepage)** and **[HEIG-VD](https://heig-vd.ch/)** as part of a **[Swiss AI Center](https://www.hes-so.ch/en/swiss-ai-center)** cheque supported by **[Exoscale](https://www.exoscale.com/)**. Explore diverse use cases while learning how to perform [supervised fine-tuning (SFT)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) of [large language models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model), including data preparation, optimization, monitoring, scaling, deployment, interaction, and evaluation.

> **Note:** This is the `website` branch containing the documentation site. The workshop code itself is on the [`main`](https://github.com/ALIRE-HES-SO/llm-sft-workshop/tree/main) branch.

## Local Development

To run the documentation site locally:

1. **Clone the repository and switch to the website branch:**
   ```bash
   git clone https://github.com/ALIRE-HES-SO/llm-sft-workshop.git
   cd llm-sft-workshop
   git checkout website
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
   Alternatively, you can prefix commands with `uv run` (e.g., `uv run zensical serve`).

4. **Start the development server:**
   ```bash
   zensical serve
   ```

5. **Open your browser** at [http://localhost:8000](http://localhost:8000)

## Project Structure

- **`docs/`** — Markdown files containing the documentation content
- **`zensical.toml`** — Main configuration file for Zensical (site settings, navigation, theme options)
