import torch
import pickle
from torch.utils.data import DataLoader
from models.classifier import ResidueClassifier as Classifier
from utils.metrics import compute_metrics, group_metrics_by_sequence
from utils.plots import plot_pr_curve, visualize_sequence_prediction

# Load test data
with open("data/test.pkl", "rb") as f:
    test_data = pickle.load(f)  # Should contain list of dicts w/ 'embedding', 'label', 'seq_id', 'sequence'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(input_dim=1024).to(device)
model.load_state_dict(torch.load("checkpoints/classifier.pt"))
model.eval()

all_y_true, all_y_pred, results = [], [], []

for item in test_data:
    emb = torch.tensor(item['embedding']).unsqueeze(0).to(device)
    label = torch.tensor(item['label']).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(emb).squeeze(0).cpu().numpy()
    y_true = label.squeeze(0).cpu().numpy()
    all_y_true.extend(y_true.tolist())
    all_y_pred.extend(pred.tolist())
    results.append((y_true, pred, item['seq_id']))

# Compute and show overall metrics
metrics = compute_metrics(all_y_true, all_y_pred)
print("Overall Metrics:", metrics)

# Save PR curve
plot_pr_curve(all_y_true, all_y_pred, save_path="outputs/pr_curve.png")

# Save sequence-level visualizations
for item, (_, pred, _) in zip(test_data, results):
    vis = visualize_sequence_prediction(item['sequence'], item['label'], pred)
    with open(f"outputs/pred_{item['seq_id']}.txt", "w") as f:
        f.write(vis + "\n")

# Grouped metrics
grouped = group_metrics_by_sequence(results)
with open("outputs/metrics_by_sequence.json", "w") as f:
    import json; json.dump(grouped, f, indent=2)

print("Evaluation complete.")
