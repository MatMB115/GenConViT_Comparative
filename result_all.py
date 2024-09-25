import os
import json
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score

json_files = [
    os.path.join("result/original", "data_april14_Celeb-DF.json"),
    os.path.join("result/original", "data_april14_DFDC.json"),
    os.path.join("result/original", "data_april11_DeepfakeTIMIT.json"),
    os.path.join("result/original", "data_april14_FF++.json"),
    os.path.join("result/original", "prediction_September_20_2024_18_44_WildDeepfake-all-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_09_11_WildDeepfake-ed-15f.json"),
    os.path.join("result/original", "prediction_September_25_2024_15_22_DeepSpeak-vae-15f.json"),
]

json_files_retrain = [
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_00_13_WildDeepfake-5e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_21_19_WildDeepfake-6e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_14_17_WildDepfake-8e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_02_47_WildDeepfake-10e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_06_26_WildDeepfake-5e-ed-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_18_42_WildDeepfake-10e-ed-15f.json"),
]

json_files_diferent_frames = [
    os.path.join("result/original", "prediction_September_24_2024_14_55_WildDeepfake-vae-10f.json"),
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/original", "prediction_September_24_2024_14_43_WildDeepfake-vae-24f.json"),
]

exec_wild_time = []
exec_wild_time_values = []
exec_dpspeak_time = []
exec_dpspeak_time_values = []

headers = ["Dataset", "Accuracy", "Real Accuracy", "Fake Accuracy", "ROC AUC", "F1 Score", "Precision", "Recall"]
print(' '.join(headers))

def extract_method(video_name):
    parts = video_name.split('--')
    if len(parts) > 0:
        return parts[0]
    return "unknown"

def autopct_format(pct, allvalues):
    absolute = int(pct/100.*sum(allvalues))
    return f"{pct:.1f}%\n({absolute} casos)"

def detect_false_positive(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    method_false_positive_count = {}
    total_false_positives = 0

    for correct_label, prediction_label, video_name in zip(data['video']['correct_label'], data['video']['pred_label'], data['video']['name']):
        if correct_label == 'FAKE' and prediction_label == 'REAL':  
            method = extract_method(video_name)
            if method not in method_false_positive_count:
                method_false_positive_count[method] = 0
            method_false_positive_count[method] += 1
            total_false_positives += 1

    method_false_positive_percentage = {method: (count / total_false_positives) * 100 for method, count in method_false_positive_count.items()}
    return method_false_positive_percentage, method_false_positive_count

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
        
        metrics_data.append([json_file[:-5].split('_')[-1].replace('vae', 'VAE').replace('ed', "ED"), f"{accuracy*100:.2f}", f"{real_acc*100:.2f}", f"{fake_acc*100:.2f}",
                             f"{roc_auc:.3f}", f"{f1:.3f}", f"{precision:.3f}", f"{recall:.3f}"])

        print(json_file[:-5].split('_')[-1].replace('vae', 'VAE').replace('ed', "ED"), f"{accuracy*100:.2f}", f"{real_acc*100:.2f}", f"{fake_acc*100:.2f}",
                             f"{roc_auc:.3f}", f"{f1:.3f}", f"{precision:.3f}", f"{recall:.3f}")

        if "WildDeepfake" in json_file:
            exec_wild_time.append(os.path.basename(json_file).split("_")[-1].replace(".json", "").replace('WildDeepfake-', ''))
            exec_wild_time_values.append(result.get("time", {}).get("elapsed", [0])[0])
        if "DeepSpeak" in json_file:
            exec_dpspeak_time.append(os.path.basename(json_file).split("_")[-1].replace(".json", "").replace('WildDeepfake-', ''))
            exec_dpspeak_time_values.append(result.get("time", {}).get("elapsed", [0])[0])

    return fpr_list, tpr_list, roc_auc_list, metrics_data

# Processing results
fpr_list, tpr_list, roc_auc_list, metrics_data = process_json_files(json_files)

fpr_list_retrain, tpr_list_retrain, roc_auc_list_retrain, metrics_data_retrain = process_json_files(json_files_retrain)

fpr_list_dif_frames, tpr_list_dif_frames, roc_auc_list_dif_frames, metrics_data_dif_frames = process_json_files(json_files_diferent_frames)

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 10))

false_positive_percentage, count = detect_false_positive(json_files[7])
df_methods = list(false_positive_percentage.keys())
fp_percentages = list(false_positive_percentage.values())
counts = [count[method] for method in df_methods]

plt.subplots_adjust(wspace=0.4)

# ROC chart of original files (first row, first column)
for i in range(len(json_files)):
    ax1.plot(fpr_list[i], tpr_list[i], label=f"{json_files[i][:-5].split('_')[-1]} (area = {roc_auc_list[i]:.3f})")

ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve - Original Models")
ax1.legend(loc="lower right", fontsize=8)

# ROC chart of retraining files (first row, second column)
for i in range(len(json_files_retrain)):
    ax2.plot(fpr_list_retrain[i], tpr_list_retrain[i], label=f"{json_files_retrain[i][:-5].split('_')[-1]} (area = {roc_auc_list_retrain[i]:.3f})")

ax2.plot([0, 1], [0, 1], "k--")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve - Retrained Models")
ax2.legend(loc="lower right", fontsize=8)

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

ax5.barh(exec_wild_time, exec_wild_time_values, color=[[random.random() for _ in range(3)] for _ in range(len(exec_wild_time))])
ax5.set_xlabel('Time (seconds)', fontsize=8)
ax5.set_ylabel('Epoch(e) - Networks - Frames(f)', fontsize=8)
ax5.set_title('Execution Time of Networks VAE and ED (WildDeepfake)')
for label in ax5.get_yticklabels():
    label.set_fontsize(8)

ax6.barh(exec_dpspeak_time, exec_dpspeak_time_values, color=[[random.random() for _ in range(3)] for _ in range(len(exec_dpspeak_time))])
ax6.set_xlabel('Time (seconds)', fontsize=8)
ax6.set_ylabel('Networks', fontsize=8)
ax6.set_title('Execution Time of Networks VAE and ED (DeepSpeak)')
for label in ax6.get_yticklabels():
    label.set_fontsize(8)

ax7.axis("off")
ax7.set_title("Sample Frames Performance Comparison", fontsize=12)
table_dif_frames = ax7.table(cellText=metrics_data_dif_frames, colLabels=headers, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 8)
table_dif_frames.auto_set_font_size(False)
table_dif_frames.auto_set_column_width(col=list(range(len(headers)))) 
table_dif_frames.set_fontsize(8)
table_dif_frames.scale(1.5, 1.5)

ax8.pie(fp_percentages, labels=df_methods, autopct=lambda pct: autopct_format(pct, counts), startangle=140, colors=plt.cm.Paired.colors)
ax8.set_title('False Positives x Method (DeepSpeak - GenConViTVAE)')
ax8.axis('equal')

plt.tight_layout()
plt.show()
