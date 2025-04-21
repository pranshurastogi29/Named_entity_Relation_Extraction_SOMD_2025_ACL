# SOMD2025 - Named Entity Recognition and Relation Extraction

This repository contains the implementation of Named Entity Recognition (NER) and Relation Extraction (RE) models using state-of-the-art transformer architectures. The project focuses on extracting structured information from text using various deep learning approaches.

## Project Overview

The project consists of several key components:
- Named Entity Recognition (NER) using DeBERTa v3
- Relation Extraction (RE) using BERT Large and DeBERTa
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
- Model: DeBERTa v3 Large
- Implementation: K-fold cross-validation (4 folds)
- Purpose: Identify and classify named entities in text

### Relation Extraction (RE)
- Models:
  - BERT Large
  - DeBERTa v3
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
- Implements both BERT Large and DeBERTa models
- Uses specialized relation extraction architecture
- Includes performance optimization techniques

## Results

Results are stored in:
- `modern_bert_large_re_results/`: BERT model performance
- `deberta_v3_re_results/`: DeBERTa model performance

## Pipeline and Submission Recreation

### Submission Recreation Process
The `pipeline_submission_nbs/submission_recreate.ipynb` notebook provides a complete pipeline to recreate our model predictions. This notebook:

1. Loads and preprocesses the input data
2. Runs the NER model (DeBERTa v3) to identify entities
3. Applies the RE models (BERT Large and DeBERTa) to extract relationships
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
- RE predictions from BERT Large and DeBERTa models
- Post-processing and optimization steps

### Important Notes
- Run the cells in sequence as specified in the notebook
- GPU is recommended for optimal performance
- Models will be downloaded automatically on first run

## Contributing

[Add contribution guidelines if applicable]

## License

[Specify license]

## Contact

[Add contact information]

## Acknowledgments

[Add acknowledgments if applicable] 