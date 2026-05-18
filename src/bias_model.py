from transformers import pipeline

# Load once, reuse
bias_pipeline = pipeline(
    "text-classification",
    model="bucketresearch/politicalBiasBERT",
    tokenizer="bucketresearch/politicalBiasBERT",
    return_all_scores=True
)

def predict_bias(text):
    truncated_text = text[:1024]
    result = bias_pipeline(truncated_text)
    
    # The model returns uppercase labels: LEFT, CENTER, RIGHT
    raw_labels = {item['label']: item['score'] for item in result[0]}
    
    return {
        "left": raw_labels.get("LEFT", 0.0),
        "center": raw_labels.get("CENTER", 0.0),
        "right": raw_labels.get("RIGHT", 0.0)
    }