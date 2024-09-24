import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score

json_files = [
    os.path.join("result/original", "data_april14_Celeb-DF.json"),
    os.path.join("result/original", "data_april14_DFDC.json"),
    os.path.join("result/original", "data_april11_DeepfakeTIMIT.json"),
    os.path.join("result/original", "data_april14_FF++.json"),
    os.path.join("result/original", "prediction_September_20_2024_18_44_02_WildDeepfake-all-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_09_11_WildDeepfake-ed-15f.json"),
]

json_files_retrain = [
    os.path.join("result/retrain", "prediction_September_24_2024_00_13_WildDeepfake-5e-vae-15f.json"),
]

headers = ["Dataset", "Accuracy", "Real Accuracy", "Fake Accuracy", "ROC AUC", "F1 Score", "Precision", "Recall"]

# Function to process JSON files and generate data for the ROC chart and statistics table
def process_json_files(json_files):
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    metrics_data = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            result = json.load(f)

        actual_labels = result["video"]["correct_label"]
        predicted_probs = result["video"]["pred"]
        predicted_labels = result["video"]["pred_label"]

        big_pp = [1 if P >= 0.5 else 0 for P in predicted_probs]
        p_labels = [1 if label == "FAKE" else 0 for label in predicted_labels]
        a_labels = [1 if label == "FAKE" else 0 for label in actual_labels]

        fpr, tpr, thresholds = roc_curve(a_labels, predicted_probs)
        roc_auc = roc_auc_score(a_labels, predicted_probs)
        f1 = f1_score(a_labels, big_pp)
        precision = precision_score(a_labels, big_pp)
        recall = recall_score(a_labels, big_pp)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        accuracy = sum(x == y for x, y in zip(p_labels, a_labels)) / len(p_labels)
        real_acc = sum((x == y and y == 0) for x, y in zip(p_labels, a_labels)) / a_labels.count(0)
        fake_acc = sum((x == y and y == 1) for x, y in zip(p_labels, a_labels)) / a_labels.count(1)
        
        metrics_data.append([json_file[:-5].split('_')[-1], f"{accuracy*100:.2f}", f"{real_acc*100:.2f}", f"{fake_acc*100:.2f}",
                             f"{roc_auc:.3f}", f"{f1:.3f}", f"{precision:.3f}", f"{recall:.3f}"])

    return fpr_list, tpr_list, roc_auc_list, metrics_data

# Processing results
fpr_list, tpr_list, roc_auc_list, metrics_data = process_json_files(json_files)

fpr_list_retrain, tpr_list_retrain, roc_auc_list_retrain, metrics_data_retrain = process_json_files(json_files_retrain)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# ROC chart of original files (first row, first column)
for i in range(len(json_files)):
    ax1.plot(fpr_list[i], tpr_list[i], label=f"{json_files[i][:-5].split('_')[-1]} (area = {roc_auc_list[i]:.3f})")

ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve - Original Models")
ax1.legend(loc="lower right")

# ROC chart of retraining files (first row, second column)
for i in range(len(json_files_retrain)):
    ax2.plot(fpr_list_retrain[i], tpr_list_retrain[i], label=f"{json_files_retrain[i][:-5].split('_')[-1]} (area = {roc_auc_list_retrain[i]:.3f})")

ax2.plot([0, 1], [0, 1], "k--")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve - Retrained Models")
ax2.legend(loc="lower right")

# Table of original files (second row, first column)
ax3.axis("off")
ax3.set_title("Original Models - Performance Metrics Comparison", fontsize=12)
table = ax3.table(cellText=metrics_data, colLabels=headers, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 8)
table.auto_set_font_size(False)
table.auto_set_column_width(col=list(range(len(headers)))) 
table.set_fontsize(8)
table.scale(1.5, 1.5)

# Table of retraining files (second row, second column)
ax4.axis("off")
ax4.set_title("Retrained Models - Performance Metrics Comparison", fontsize=12)
table_retrain = ax4.table(cellText=metrics_data_retrain, colLabels=headers, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 8)
table_retrain.auto_set_font_size(False)
table_retrain.auto_set_column_width(col=list(range(len(headers)))) 
table_retrain.set_fontsize(8)
table_retrain.scale(1.5, 1.5)

plt.tight_layout()
plt.show()
