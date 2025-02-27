# Product Specification NER Solution

This project implements a Named Entity Recognition (NER) solution for automatically detecting product specifications from unstructured text. It focuses on extracting three key attributes:

1. **Brand** - The manufacturer or brand name (e.g., Samsung, Western Digital, Lenovo)
2. **Storage Capacity (Speicherkapazität)** - The storage capacity of the product (e.g., 512 GB, 2 TB)
3. **Color (Farbe)** - The color of the product (e.g., Schwarz, Weiß, Blau)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Using the API](#using-the-api)
  - [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Enhancing Sellers' Experience](#enhancing-sellers-experience)
- [Data Preparation Improvements](#data-preparation-improvements)
- [License](#license)

## Overview

This project creates a machine learning solution that extracts product specifications from unstructured text. It uses SpaCy's NER capabilities to identify and label entities in German text. The solution is packaged as a Dockerized REST API that can be easily deployed and integrated with other systems.

## Project Structure

```
.
├── app.py                  # FastAPI application
├── product_ner.py          # Core NER implementation
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── data/                   # Directory for training data
├── model/                  # Directory for trained models
├── tests/                  # Unit and integration tests
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- At least 4GB RAM for model training

### Installation

#### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd product-ner
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the German language model for SpaCy:
```bash
python -m spacy download de_core_news_sm
```

#### Docker Installation

Simply run:
```bash
docker-compose up -d
```

This will build the Docker image and start the API service.

## Usage

### Training the Model

To train the NER model with your data:

```bash
python product_ner.py
```

This will:
1. Preprocess the data from `ds_ner_test_case.csv`
2. Train a new NER model
3. Evaluate the model's performance
4. Save the model to the `./model` directory

### Using the API

Once the API is running (either locally or via Docker), you can use it to extract product specifications from text:

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Samsung Galaxy S21 Ultra mit 512 GB Speicher in Phantom Black"}'
```

#### Example Response

```json
{
  "text": "Samsung Galaxy S21 Ultra mit 512 GB Speicher in Phantom Black",
  "entities": [
    {
      "text": "Samsung",
      "label": "BRAND",
      "start": 0,
      "end": 7
    },
    {
      "text": "512 GB",
      "label": "STORAGE",
      "start": 24,
      "end": 30
    },
    {
      "text": "Phantom Black",
      "label": "COLOR",
      "start": 42,
      "end": 55
    }
  ],
  "tagged_text": "[Samsung](BRAND) Galaxy S21 Ultra mit [512 GB](STORAGE) Speicher in [Phantom Black](COLOR)"
}
```

### Docker Deployment

To deploy using Docker:

1. Build and start the services:
```bash
docker-compose up -d
```

2. Check that the service is running:
```bash
docker-compose ps
```

3. View the logs:
```bash
docker-compose logs -f
```

4. Stop the services:
```bash
docker-compose down
```

## API Documentation

When the API is running, you can access the automatically generated Swagger UI documentation at:

```
http://localhost:8000/docs
```

This provides an interactive interface to test the API endpoints and view the API documentation.

## Testing

### Running Unit Tests

```bash
pytest tests/test_product_ner.py -v
```

### Running Integration Tests

```bash
pytest tests/test_integration.py -v
```

## Enhancing Sellers' Experience

This solution significantly enhances the sellers' experience in onboarding new products by:

1. **Automated Specification Extraction**: Sellers no longer need to manually input all specifications, reducing data entry time.

2. **Reduced Errors**: Automatic extraction ensures consistent terminology and reduces human error.

3. **Improved Efficiency**: The time to onboard new products is dramatically reduced, allowing sellers to list more products faster.

4. **Better Data Quality**: Standardized extraction of key specifications ensures data consistency across the marketplace.

5. **Streamlined Workflow**: Sellers can focus on providing unique product descriptions rather than basic specifications.

6. **Reduced Support Needs**: Fewer manual entries means fewer support tickets for fixing data entry errors.

7. **Cross-Checking Capability**: The system can verify entered specifications against those extracted from descriptions, highlighting potential inconsistencies.

## Data Preparation Improvements

The current data preparation uses a simple string matching approach. Here are ways to improve it:

1. **Entity Resolution and Normalization**: Implement techniques to recognize different variations of the same entity (e.g., "WD" and "Western Digital" as the same brand).

2. **Contextual Extraction**: Use the surrounding context to improve entity extraction accuracy.

3. **Regex Pattern Enhancement**: Develop more sophisticated patterns for capturing complex specifications.

4. **Rule-Based Preprocessing**: Add domain-specific rules to handle special cases in product descriptions.

5. **Active Learning Pipeline**: Implement a feedback loop where incorrect predictions can be corrected to improve the model over time.

6. **Multi-language Support**: Extend the model to support multiple languages to accommodate international sellers.

7. **Custom Entity Linking**: Link extracted entities to a knowledge base of brands, standard storage sizes, and colors.

8. **Distant Supervision**: Use existing product databases to automatically label more training data.

9. **Data Augmentation**: Generate synthetic variations of product descriptions to improve model robustness.

10. **Hybrid Approach**: Combine the statistical NER model with rule-based approaches for best results.

## License

[Specify License]
