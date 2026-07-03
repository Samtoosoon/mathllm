# Math LLM for Hindi Medium Students 🧮🇮🇳

An industry-grade AI tutor designed to solve and explain mathematical problems in **Hindi (Devanagari)** for students from Classes 6–10. The system leverages **Mistral-7B-Instruct** fine-tuned with **QLoRA** on a translated subset of the MetaMathQA dataset.

## Architecture

```mermaid
graph TD
    A[User (Streamlit UI)] -->|Hindi Math Query| B(FastAPI Backend)
    B --> C{PeftModel + Base Mistral 7B}
    C -->|Generates Steps| B
    B -->|JSON Response| A
    
    D[MetaMathQA] -->|deep-translator| E[Hindi MetaMathQA]
    E -->|SFTTrainer + bitsandbytes| C
```

## Features
- **Accurate Mathematical Reasoning**: Fine-tuned to think step-by-step.
- **Hindi Explanations**: Output is generated natively in Devanagari.
- **Production-Ready API**: FastAPI backend with easy-to-use `/solve` endpoint.
- **Intuitive UI**: Built with Streamlit for a seamless tutoring experience.
- **Optimized Deployment**: Distributed via Docker and Docker Compose.

## Repository Structure
```
math-llm/
├── api/             # FastAPI inference application
├── configs/         # QLoRA and training hyperparameters
├── datasets/        # Scripts for downloading and translating datasets
├── docker/          # Dockerfiles and Compose configurations
├── evaluation/      # Metric calculation (ROUGE, BLEU, Exact Match)
├── frontend/        # Streamlit user interface
└── training/        # Model fine-tuning logic (PEFT/TRL)
```

## Setup & Installation

### 1. Dataset Preparation
Translate the dataset into Hindi:
```bash
pip install -r api/requirements.txt deep-translator datasets
cd datasets
python prepare_dataset.py --sample_size 10000
```

### 2. Training (Requires GPU > 16GB VRAM)
```bash
cd training
python train.py --config ../configs/qlora_config.yaml
```

### 3. Running the Stack (Docker)
Ensure you have Docker and the NVIDIA Container Toolkit installed.
```bash
cd docker
docker-compose up --build
```
- Frontend UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Evaluation Metrics
We evaluate the fine-tuned model against the base model using:
- **Exact Match** (Heuristic parsing of the final mathematical answer)
- **ROUGE** (Reasoning step overlap)
- **BLEU** (Linguistic translation quality)

Run the evaluation:
```bash
cd evaluation
python evaluate.py --num_samples 100
```

---
*Built with ❤️ for Software Engineering and AI.*
