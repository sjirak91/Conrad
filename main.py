import json
from src.product_ner import ProductNER

def main():
    """Main function to run the NER process"""
    # Initialize the ProductNER class
    product_ner = ProductNER()
    
    # Preprocess the data
    data = product_ner.preprocess_data("ds_ner_test_case.csv")
    
    # Train the model
    model_path = product_ner.train_model("./data/train.spacy")
    
    # Evaluate the model
    scores = product_ner.evaluate_model("./data/test.spacy")
    
    print("Model evaluation results:")
    print(f"Overall precision: {scores['overall']['precision']:.4f}")
    print(f"Overall recall: {scores['overall']['recall']:.4f}")
    print(f"Overall F-score: {scores['overall']['f_score']:.4f}")
    
    # Save detailed results to a JSON file
    with open("./model/evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    # Example usage
    example_text = "Samsung Galaxy S21 Ultra mit 512 GB Speicher in Phantom Black"
    entities = product_ner.predict(example_text)
    
    print("\nExample prediction:")
    print(f"Text: {example_text}")
    print("Entities:", entities)
    
    tagged_text = product_ner.tag_text(example_text)
    print(f"Tagged text: {tagged_text}")

if __name__ == "__main__":
    main() 