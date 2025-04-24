# SOMD2025 - Named Entity Recognition and Relation Extraction

As part of the SOMD 2025 shared task on Software Mention Detection, we solved the problem of detecting and disambiguating software mentions in academic texts. a very important but under appreciated factor in research transparency and reproducibility. Software is an essential building block of scientific activity, but it often does not receive official citation in scholarly literature, and there are many informal mentions that are hard to follow and analyse. In order to enhance research accessibility and interpretability, we built a system that identifies software mentions and their properties (e.g., version numbers, URLs) as named entities, and classify relationships between them. Our dataset contained approximately 1,100 manually annotated sentences of full-text scholarly articles, representing diverse types of software like operating systems and applications. We fine-tuned DeBERTa based models for the Named Entity Recognition (NER) task and handled Relation Extraction (RE) as a classification problem over entity pairs. Due to the dataset size, we employed Large Language Models to create synthetic training data for augmentation. Our system achieved strong performance, with a 65% F1 score on NER (ranking 2nd in test phase) and a 47% F1 score on RE and combined macro 56% F1, showing the performance of our approach in this area.

## Project Overview

The project consists of several key components:
- Named Entity Recognition (NER) using DeBERTa v3
- Relation Extraction (RE) using ModernBERT Large and DeBERTa
- Synthetic dataset generation using Large Language Models (LLMs)

## Directory Structure

```
.
├── pipeline_submission_nbs/     # Pipeline submission notebooks
├── synthetic_dataset/          # Generated synthetic dataset
├── named_entity_recognition_model/  # NER model implementations
│   ├── deberta-v3-large-NER-Fold_0.ipynb
│   ├── deberta-v3-large-NER-Fold_1.ipynb
│   ├── deberta-v3-large-NER-Fold_2.ipynb
│   └── deberta-v3-large-NER-Fold_3.ipynb
├── relation_extraction_models/  # RE model implementations
│   ├── Relation_extraction_deberta_fold_2.ipynb
│   └── Relation_extraction_Modern_bert_large_fold_0.ipynb
├── llm_generate_data_nbs/      # Notebooks for synthetic data generation
├── SOMD2025-full_data/        # Full dataset directory
├── modern_bert_large_re_results/  # BERT model results
├── deberta_v3_re_results/     # DeBERTa model results
├── relations.csv              # Relations dataset
└── relations_test.csv         # Test dataset
```

## Models

### Named Entity Recognition (NER)
- Model: DeBERTa v3 Large - psresearch/deberta-v3-large-NER-Scholarly-text\href{https://huggingface.co/psresearch/deberta-v3-large-NER-Scholarly-text}
- Implementation: K-fold cross-validation (4 folds)
- Purpose: Identify and classify named entities in text

### Relation Extraction (RE)
- Models:
  - ModernBERT Large
  - DeBERTa v3 - 
- Purpose: Extract relationships between identified entities

## Data Generation

The project includes synthetic data generation using various LLMs:
- Mistral 7B
- Qwen 2.5 7B
- Gemma 2 9B

## Getting Started

1. Clone the repository:
```bash
git clone [repository-url]
cd SOMD2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights:
```bash
# The models will be downloaded automatically when running the notebooks
```

4. Run the notebooks:
- For NER: Use notebooks in `named_entity_recognition_model/`
- For RE: Use notebooks in `relation_extraction_models/`
- For data generation: Use notebooks in `llm_generate_data_nbs/`

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for Python package dependencies

## Model Training

### NER Training
- Uses DeBERTa v3 Large with 4-fold cross-validation
- Includes data augmentation and model merging techniques
- Implements custom token classification head

### RE Training
- Implements both ModernBERT Large and DeBERTa models
- Uses specialized relation extraction architecture
- Includes performance optimization techniques

## Results

Results are stored in:
- `modern_bert_large_re_results/`: ModernBERT model performance
- `deberta_v3_re_results/`: DeBERTa model performance

## Pipeline and Submission Recreation

### Submission Recreation Process
The `pipeline_submission_nbs/submission_recreate.ipynb` notebook provides a complete pipeline to recreate our model predictions. This notebook:

1. Loads and preprocesses the input data
2. Runs the NER model (DeBERTa v3) to identify entities
3. Applies the RE models (ModernBERT Large and DeBERTa) to extract relationships
4. Combines and post-processes the results

To recreate the submission:

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Open `pipeline_submission_nbs/submission_recreate.ipynb`

3. Run all cells in sequence to:
   - Load the required models
   - Process the test dataset
   - Generate final predictions

### Pipeline Components
The submission pipeline integrates:
- NER model predictions from 4-fold DeBERTa v3 models
- RE predictions from ModernBERT Large and DeBERTa models
- Post-processing and optimization steps

### Important Notes
- Run the cells in sequence as specified in the notebook
- GPU is recommended for optimal performance
- Models will be downloaded automatically on first run

## License

MIT License 
