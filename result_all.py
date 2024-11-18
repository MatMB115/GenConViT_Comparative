import os
import re
import json
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.ticker as ticker

output_dir = "result/assets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

json_files = [
    os.path.join("result/original", "data_april14_Celeb-DF.json"),
    os.path.join("result/original", "data_april14_DFDC.json"),
    os.path.join("result/original", "data_april11_DeepfakeTIMIT.json"),
    os.path.join("result/original", "data_april14_FF++.json"),
    # os.path.join("result/original", "prediction_September_20_2024_18_44_WildDeepfake-all-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/original", "prediction_September_21_2024_09_11_WildDeepfake-ed-15f.json"),
    os.path.join("result/original", "prediction_September_25_2024_15_22_DeepSpeak-vae-15f.json"),
    os.path.join("result/original", "prediction_September_26_2024_04_30_DeepSpeak-ed-15f.json"),
]

json_files_retrain_wild = [
    os.path.join("result/retrain", "prediction_September_26_2024_01_15_WildDeepfake-4e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_00_13_WildDeepfake-5e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_21_19_WildDeepfake-6e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_14_17_WildDeepfake-8e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_02_47_WildDeepfake-10e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_06_26_WildDeepfake-5e-ed-15f.json"),
    os.path.join("result/retrain", "prediction_September_24_2024_18_42_WildDeepfake-10e-ed-15f.json"),
]

json_files_retrain_speak = [
    os.path.join("result/retrain", "prediction_October_03_2024_15_59_DeepSpeak-4e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_00_01_DeepSpeak-5e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_05_43_DeepSpeak-8e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_October_03_2024_10_27_DeepSpeak-4e-ed-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_11_44_DeepSpeak-5e-ed-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_22_21_DeepSpeak-8e-ed-15f.json"),
]

json_files_diferent_frames = [
    os.path.join("result/original", "prediction_September_24_2024_14_55_WildDeepfake-vae-10f.json"),
    os.path.join("result/original", "prediction_September_21_2024_03_19_WildDeepfake-vae-15f.json"),
    os.path.join("result/original", "prediction_September_24_2024_14_43_WildDeepfake-vae-24f.json"),
]

json_files_deepspeak =[
    os.path.join("result/original", "prediction_September_25_2024_15_22_DeepSpeak-vae-15f.json"),
    os.path.join("result/original", "prediction_September_26_2024_04_30_DeepSpeak-ed-15f.json"),
]

json_files_retrain_deepspeak =[
    #os.path.join("result/retrain", "prediction_October_03_2024_10_27_DeepSpeak-4e-ed-15f.json"),
    #reg os.path.join("result/retrain", "prediction_October_03_2024_15_59_DeepSpeak-4e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_00_01_DeepSpeak-5e-vae-15f.json"),
    os.path.join("result/retrain", "prediction_October_10_2024_11_44_DeepSpeak-5e-ed-15f.json"),
]

pickle_files = [
    os.path.join("result/original_benchmark", "xception_DeepSpeak_metrics.pkl"),
    os.path.join("result/original_benchmark", "efficientnetb4_DeepSpeak_metrics.pkl"),
    os.path.join("result/original_benchmark", "meso4Inception_DeepSpeak_metrics.pkl"),
    os.path.join("result/original_benchmark", "spsl_DeepSpeak_metrics.pkl"),
    os.path.join("result/original_benchmark", "ucf_DeepSpeak_metrics.pkl"),
]

pickle_files_retrain_deepspeak = [
    os.path.join("result/retrain_benchmark", "xception_DeepSpeak_metrics_20241023_211810.pkl"),
    os.path.join("result/retrain_benchmark", "efficientnetb4_DeepSpeak_metrics_20241023_153344.pkl"),
    os.path.join("result/retrain_benchmark", "meso4Inception_DeepSpeak_metrics_20241026_164837.pkl"),
    os.path.join("result/retrain_benchmark", "spsl_DeepSpeak_metrics_20241023_041542.pkl"),
    os.path.join("result/retrain_benchmark", "ucf_DeepSpeak_metrics_20241027_022453.pkl"),
]

exec_wild_time = []
exec_wild_time_values = []
exec_dpspeak_time = []
exec_dpspeak_time_values = []

default_header = ["Acurácia", "Acurácia Real", "Acurácia Fake", "ROC AUC", "F1 Score", "Precisão", "Recall"]
headers = ["Dataset"] + default_header
headers_bench = ["Arquitetura"] + default_header
headers_retrain = ["Rede", "Época"] + default_header
headers_dif_frames = ["Dataset", "Rede", "Frames"] + default_header

def save_axis(ax, filename, fig):
    for axis in fig.axes:
        if axis != ax:
            axis.set_visible(False)

    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300, format='pdf')

    for axis in fig.axes:
        if axis != ax:
            axis.set_visible(True)

def save_table(data, columns, filename):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(output_dir, filename + ".csv"), index=False)

def extract_method(video_name):
    parts = video_name.split('--')
    if len(parts) > 0:
        return parts[0]
    return "unknown"

def autopct_format(pct, allvalues):
    absolute = int(pct/100.*sum(allvalues))
    return f"{pct:.1f}%\n({absolute})".replace('.', ',')

def format_decimal(x, pos):
    return f'{x:.2f}'.replace('.', ',')

def format_table_decimal(value):
    return f"{value}".replace('.', ',')

def detect_false_negative(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    method_false_negative_count = {}
    total_false_negatives = 0

    for correct_label, prediction_label, video_name in zip(data['video']['correct_label'], data['video']['pred_label'], data['video']['name']):
        if correct_label == 'FAKE' and prediction_label == 'REAL':  
            method = extract_method(video_name)
            if method not in method_false_negative_count:
                method_false_negative_count[method] = 0
            method_false_negative_count[method] += 1
            total_false_negatives += 1

    method_false_positive_percentage = {method: (count / total_false_negatives) * 100 for method, count in method_false_negative_count.items()}
    return method_false_positive_percentage, method_false_negative_count

def process_files(json_files, pickle_files=[], is_genconvit=False, is_dif_frames=False, is_benchmark=False):
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
        
        match = re.search(r'-(vae|ed)-', json_file)
        if match:
            match = match.group(1).upper()
        else:
            match = ""
        if is_genconvit:
            metrics_data.append([match, json_file[:-5].split('_')[-1].split('-')[1].replace('e', ''), format_table_decimal(f"{accuracy*100:.2f}"), 
                                 format_table_decimal(f"{real_acc*100:.2f}"), format_table_decimal(f"{fake_acc*100:.2f}"),
                                format_table_decimal(f"{roc_auc:.3f}"), format_table_decimal(f"{f1:.3f}"), format_table_decimal(f"{precision:.3f}"), 
                                format_table_decimal(f"{recall:.3f}")])
        elif is_dif_frames:
                metrics_data.append([json_file[:-9].split('_')[-1].replace('-vae', '').replace('-ed', ''), match, json_file[-8:-5], 
                                     format_table_decimal(f"{accuracy*100:.2f}"), format_table_decimal(f"{real_acc*100:.2f}"), 
                                     format_table_decimal(f"{fake_acc*100:.2f}"),
                                format_table_decimal(f"{roc_auc:.3f}"), format_table_decimal(f"{f1:.3f}"), 
                                format_table_decimal(f"{precision:.3f}"), format_table_decimal(f"{recall:.3f}")])
        elif is_benchmark:
            metrics_data.append([json_file[:-5].split('_')[-1].replace('DeepSpeak', 'GenConViT').replace('-15f', '').replace('ed', 'ae').upper(), 
                                 format_table_decimal(f"{accuracy*100:.2f}"), format_table_decimal(f"{real_acc*100:.2f}"), 
                                 format_table_decimal(f"{fake_acc*100:.2f}"),
                                format_table_decimal(f"{roc_auc:.3f}"), format_table_decimal(f"{f1:.3f}"), 
                                format_table_decimal(f"{precision:.3f}"), format_table_decimal(f"{recall:.3f}")])
        else:
            metrics_data.append([json_file[:-5].split('_')[-1].replace('vae', 'VAE').replace('ed', "ED"), 
                                 format_table_decimal(f"{accuracy*100:.2f}"), format_table_decimal(f"{real_acc*100:.2f}"), 
                                 format_table_decimal(f"{fake_acc*100:.2f}"),
                                format_table_decimal(f"{roc_auc:.3f}"), format_table_decimal(f"{f1:.3f}"), 
                                format_table_decimal(f"{precision:.3f}"), format_table_decimal(f"{recall:.3f}")])

        if "WildDeepfake" in json_file and not "all" in json_file:
            exec_wild_time.append(os.path.basename(json_file).split("_")[-1].replace(".json", "").replace('WildDeepfake-', '').replace('vae', 'VAE').replace('ed', "AE").replace('-', ' - '))
            exec_wild_time_values.append(result.get("time", {}).get("elapsed", [0])[0])
        if "DeepSpeak" in json_file:
            exec_dpspeak_time.append(os.path.basename(json_file).split("_")[-1].replace(".json", "").replace('DeepSpeak-', '').replace('vae', 'VAE').replace('ed', "AE").replace('-', ' - '))
            exec_dpspeak_time_values.append(result.get("time", {}).get("elapsed", [0])[0])
    
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            result = pickle.load(f)
        
        # Check the values generated after prediction to compare with calculated
        #print(f"Valores do pickle {pickle_file}:")
        #for key, value in result.items():
        #    if not isinstance(value, np.ndarray):
        #        print(f"{key}: {value}")
        #print("\n")
        
        predicted_probs = result['pred']
        actual_labels = result['label']

        big_pp = [1 if P >= 0.5 else 0 for P in predicted_probs]
        
        total_real = np.sum(np.array(actual_labels) == 0)
        total_fake = np.sum(np.array(actual_labels) == 1)
        real_correct = np.sum((np.array(big_pp) == 0) & (np.array(actual_labels) == 0))
        fake_correct = np.sum((np.array(big_pp) == 1) & (np.array(actual_labels) == 1))
        
        fpr, tpr, thresholds = roc_curve(actual_labels, predicted_probs)
        roc_auc = roc_auc_score(actual_labels, predicted_probs)
        f1 = f1_score(actual_labels, big_pp)
        precision = precision_score(actual_labels, big_pp)
        recall = recall_score(actual_labels, big_pp)
        accuracy = np.mean(big_pp == actual_labels)
        real_acc = real_correct / total_real if total_real > 0 else 0
        fake_acc = fake_correct / total_fake if total_fake > 0 else 0
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        metrics_data.append([pickle_file[:-4].split('/')[2].split('_')[0].upper(), format_table_decimal(f"{accuracy*100:.2f}"), 
                            format_table_decimal(f"{real_acc*100:.2f}"), format_table_decimal(f"{fake_acc*100:.2f}"), format_table_decimal(f"{roc_auc:.3f}"), 
                            format_table_decimal(f"{f1:.3f}"), format_table_decimal(f"{precision:.3f}"), format_table_decimal(f"{recall:.3f}")])
        
    return fpr_list, tpr_list, roc_auc_list, metrics_data

# Processing results
fpr_list, tpr_list, roc_auc_list, metrics_data = process_files(json_files)

fpr_list_retrain_wild, tpr_list_retrain_wild, roc_auc_list_retrain_wild, metrics_data_retrain_wild = process_files(json_files_retrain_wild, is_genconvit=True)

fpr_list_retrain_speak, tpr_list_retrain_speak, roc_auc_list_retrain_speak, metrics_data_retrain_speak = process_files(json_files_retrain_speak, is_genconvit=True)

fpr_list_dif_frames, tpr_list_dif_frames, roc_auc_list_dif_frames, metrics_data_dif_frames = process_files(json_files_diferent_frames, is_dif_frames=True)

fpr_list_benchmark, tpr_list_benchmark, roc_auc_list_benchmark, metrics_data_benchmark = process_files(json_files_deepspeak,pickle_files, is_benchmark=True)

fpr_list_benchmark_retrain, tpr_list_benchmark_retrain, roc_auc_list_benchmark_retrain, metrics_data_benchmark_retrain = process_files(json_files_retrain_deepspeak,pickle_files_retrain_deepspeak, is_benchmark=True)
# Define figures and axes

fig_roc, ((ax1, ax2), (ax11, ax13)) = plt.subplots(2, 2, figsize=(15, 10))

fig_exec_time, ((ax5, ax6), (ax10, ax14)) = plt.subplots(2, 2, figsize=(15, 10))

fig_tables, ((ax3, ax4), (ax7, ax9), (ax12, ax15)) = plt.subplots(3, 2, figsize=(20, 10))

fig_pie, ((ax8, ax10), (ax16, ax16)) = plt.subplots(2, 2, figsize=(20, 10))

false_positive_percentage_vae, count = detect_false_negative(json_files[6])
df_methods_vae = list(false_positive_percentage_vae.keys())
fp_percentages_vae = list(false_positive_percentage_vae.values())
counts = [count[method] for method in df_methods_vae]

false_positive_percentage_ed, count = detect_false_negative(json_files[7])
df_methods_ed = list(false_positive_percentage_ed.keys())
fp_percentages_ed = list(false_positive_percentage_ed.values())
counts = [count[method] for method in df_methods_ed]

plt.subplots_adjust(wspace=0.3, hspace=0.01)

for i in range(len(json_files)):
    ax1.plot(fpr_list[i], tpr_list[i], label=f"{json_files[i][:-5].split('_')[-1].replace('-15f', '').replace('-vae', ' - VAE').replace('-ed', ' - AE')} (área = {roc_auc_list[i]:.3f})".replace('.', ','))

ax1.plot([0, 1], [0, 1], "k--")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("Taxa de Falsos Positivos")
ax1.set_ylabel("Taxa de Verdadeiros Positivos")
ax1.set_title("Curva ROC - Modelos Originais")
ax1.legend(loc="lower right", fontsize=8)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

for i in range(len(json_files_retrain_wild)):
    ax2.plot(fpr_list_retrain_wild[i], tpr_list_retrain_wild[i], 
             label=f"{json_files_retrain_wild[i][:-5].split('_')[-1].replace('WildDeepfake-', '').replace('-15f','').replace('-ed', ' - AE').replace('-',' - ').upper()} (área = {roc_auc_list_retrain_wild[i]:.3f})".replace('.', ','))

ax2.plot([0, 1], [0, 1], "k--")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Taxa de Falsos Positivos")
ax2.set_ylabel("Taxa de Verdadeiros Positivos")
ax2.set_title("Curva ROC - Desempenho em Diferentes Épocas para o WildDeepfake")

handles, labels = ax2.get_legend_handles_labels()
handles.insert(0, plt.Line2D([0], [0], color='w', label='Épocas - Rede'))
ax2.legend(handles=handles, loc="lower right", fontsize=8)

ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

for i in range(len(json_files_retrain_wild)):
    ax11.plot(fpr_list_benchmark[i], tpr_list_benchmark[i], label=f"{metrics_data_benchmark[i][0]} (área = {roc_auc_list_benchmark[i]:.3f})".replace('.', ','))

ax11.plot([0, 1], [0, 1], "k--")
ax11.set_xlim([0.0, 1.0])
ax11.set_ylim([0.0, 1.05])
ax11.set_xlabel("Taxa de Falsos Positivos")
ax11.set_ylabel("Taxa de Verdadeiros Positivos")
ax11.set_title("Curva ROC - Comparação dos Modelos Originais")
ax11.legend(loc="lower right", fontsize=8)
ax11.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))
ax11.yaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

for i in range(len(json_files_retrain_speak)):
    ax13.plot(fpr_list_retrain_speak[i], tpr_list_retrain_speak[i], 
              label=f"{json_files_retrain_speak[i][:-5].split('_')[-1].replace('DeepSpeak-', '').replace('-15f','').replace('-ed', ' - ae').replace('-',' - ').upper()} (área = {roc_auc_list_retrain_speak[i]:.3f})".replace('.', ','))

ax13.plot([0, 1], [0, 1], "k--")
ax13.set_xlim([0.0, 1.0])
ax13.set_ylim([0.0, 1.05])
ax13.set_xlabel("Taxa de Falsos Positivos")
ax13.set_ylabel("Taxa de Verdadeiros Positivos")
ax13.set_title("Curva ROC - Desempenho em Diferentes Épocas para o DeepSpeak")

handles, labels = ax13.get_legend_handles_labels()
handles.insert(0, plt.Line2D([0], [0], color='w', label='Épocas - Rede'))
ax13.legend(handles=handles, loc="lower right", fontsize=8)

ax13.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))
ax13.yaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

for i in range(len(metrics_data_benchmark_retrain)):
    ax14.plot(fpr_list_benchmark_retrain[i], tpr_list_benchmark_retrain[i], label=f"{metrics_data_benchmark_retrain[i][0]} (área = {roc_auc_list_benchmark_retrain[i]:.3f})".replace('.', ','))

ax14.plot([0, 1], [0, 1], "k--")
ax14.set_xlim([0.0, 1.0])
ax14.set_ylim([0.0, 1.05])
ax14.set_xlabel("Taxa de Falsos Positivos")
ax14.set_ylabel("Taxa de Verdadeiros Positivos")
ax14.set_title("Curva ROC - Desempenho dos Modelos Ajustados no DeepSpeak")
ax14.legend(loc="lower right", fontsize=8)
ax14.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))
ax14.yaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

ax3.axis("off")
ax3.set_title("Comparação de Métricas - Modelos Originais", fontsize=12)
table = ax3.table(cellText=metrics_data, colLabels=headers, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 8)
table.auto_set_font_size(False)
table.auto_set_column_width(col=list(range(len(headers)))) 
table.set_fontsize(8)
table.scale(1.5, 1.5)

ax4.axis("off")
ax4.set_title("GenConViT Reajustado no WildDeepfake", fontsize=12)
table_retrain = ax4.table(cellText=metrics_data_retrain_wild, colLabels=headers_retrain, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 9)
table_retrain.auto_set_font_size(False)
table_retrain.auto_set_column_width(col=list(range(len(headers)))) 
table_retrain.set_fontsize(8)
table_retrain.scale(1.5, 1.5)

ax7.axis("off")
ax7.set_title("Comparação de Desempenho - Diferentes Frames", fontsize=12)
table_dif_frames = ax7.table(cellText=metrics_data_dif_frames, colLabels=headers_dif_frames, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 10)
table_dif_frames.auto_set_font_size(False)
table_dif_frames.auto_set_column_width(col=list(range(len(headers)))) 
table_dif_frames.set_fontsize(8)
table_dif_frames.scale(1.5, 1.5)

ax9.axis('off')
ax9.set_title("GenConViT Reajustado no DeepSpeak", fontsize=12)
table_retrain_speak = ax9.table(cellText=metrics_data_retrain_speak, colLabels=headers_retrain, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 9)
table_retrain_speak.auto_set_font_size(False)
table_retrain_speak.auto_set_column_width(col=list(range(len(headers)))) 
table_retrain_speak.set_fontsize(8)
table_retrain_speak.scale(1.5, 1.5)

ax12.axis("off")
ax12.set_title("Comparação de Métricas - Modelos Originais", fontsize=12)
table = ax12.table(cellText=metrics_data_benchmark, colLabels=headers_bench, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 9)
table.auto_set_font_size(False)
table.auto_set_column_width(col=list(range(len(headers_bench)))) 
table.set_fontsize(8)
table.scale(1.5, 1.5)

ax15.axis("off")
ax15.set_title("Comparação de Métricas - Modelos Ajustados", fontsize=12)
table = ax15.table(cellText=metrics_data_benchmark_retrain, colLabels=headers_bench, cellLoc='center', loc='center', colColours=["#c1c0c0"] * 9)
table.auto_set_font_size(False)
table.auto_set_column_width(col=list(range(len(headers_bench)))) 
table.set_fontsize(8)
table.scale(1.5, 1.5)

exec_wild_time_sorted, exec_wild_time_values_sorted = zip(*sorted(zip(exec_wild_time, exec_wild_time_values), key=lambda x: x[1]))
exec_dpspeak_time_sorted, exec_dpspeak_time_values_sorted = zip(*sorted(zip(exec_dpspeak_time, exec_dpspeak_time_values), key=lambda x: x[1]))

ax5.barh(exec_wild_time_sorted, exec_wild_time_values_sorted, color=plt.cm.Paired.colors)  # type: ignore
ax5.set_xlabel('Tempo (segundos)', fontsize=8)
ax5.set_ylabel('Época(e) - Redes - Frames(f)', fontsize=8)
ax5.set_title('Tempo de Execução das Redes VAE e AE (WildDeepfake)')
for label in ax5.get_yticklabels():
    label.set_fontsize(8)

ax6.barh(exec_dpspeak_time_sorted, exec_dpspeak_time_values_sorted, color=plt.cm.Paired.colors)  # type: ignore
ax6.set_xlabel('Tempo (segundos)', fontsize=8)
ax6.set_ylabel('Época(e) - Redes - Frames(f)', fontsize=8)
ax6.set_title('Tempo de Execução das Redes VAE e AE (DeepSpeak)')
for label in ax6.get_yticklabels():
    label.set_fontsize(8)

ax8.pie(fp_percentages_vae, labels=df_methods_vae, autopct=lambda pct: autopct_format(pct, counts), startangle=140, colors=plt.cm.Paired.colors)   # type: ignore
ax8.set_title('Falsos Negativos x Método (DeepSpeak - GenConViT - VAE)')
ax8.axis('equal')

ax10.pie(fp_percentages_ed, labels=df_methods_ed, autopct=lambda pct: autopct_format(pct, counts), startangle=140, colors=plt.cm.Paired.colors)   # type: ignore
ax10.set_title('Falsos Negativos x Método (DeepSpeak - GenConViT - AE)')
ax10.axis('equal')

save_table(metrics_data, headers, "tabela_metricas_modelos_originais")
save_table(metrics_data_retrain_wild, headers_retrain, "tabela_metricas_modelos_retrain_wild")
save_table(metrics_data_dif_frames, headers_dif_frames, "tabela_metricas_modelos_dif_frames")
save_table(metrics_data_retrain_speak, headers_retrain, "tabela_metricas_modelos_retrain_speak")

plt.tight_layout()
save_axis(ax1, "curva_roc_modelos_originais.pdf", fig_roc)
save_axis(ax2, "curva_roc_modelos_retrain_wild.pdf", fig_roc)
save_axis(ax11, "curva_roc_modelos_benchmark.pdf", fig_roc)
save_axis(ax13, "curva_roc_modelos_retrain_speak.pdf", fig_roc)
save_axis(ax14, "curva_roc_modelos_benchmark_retrain.pdf", fig_exec_time)
save_axis(ax3, "tabela_metricas_modelos_originais.pdf", fig_tables)
save_axis(ax4, "tabela_metricas_modelos_retrain_wild.pdf", fig_tables)
save_axis(ax7, "tabela_metricas_modelos_dif_frames.pdf", fig_tables)
save_axis(ax9, "tabela_metricas_modelos_retrain_speak.pdf", fig_tables)
save_axis(ax12, "tabela_metricas_modelos_benchmark.pdf", fig_tables)
save_axis(ax15, "tabela_metricas_modelos_benchmark_retrain.pdf", fig_tables)
save_axis(ax5, "tempo_execucao_wild.pdf", fig_exec_time)
save_axis(ax6, "tempo_execucao_dpspeak.pdf", fig_exec_time)
save_axis(ax8, "falsos_negativos_vae.pdf", fig_pie)
save_axis(ax10, "falsos_negativos_ed.pdf", fig_pie)
plt.show()

df_original_metrics_genconvit = pd.DataFrame(metrics_data, columns=headers)
#df_original_metrics_genconvit.to_csv(os.path.join(output_dir, "original_genconvit_metrics.csv"), index=False)

df_retrain_metrics_genconvit_wild = pd.DataFrame(metrics_data_retrain_wild, columns=headers_retrain)
#df_retrain_metrics_genconvit_wild.to_csv(os.path.join(output_dir, "retrain_metrics_genconvit_wild.csv"), index=False)

df_retrain_metrics_genconvit_speak = pd.DataFrame(metrics_data_retrain_speak, columns=headers_retrain)
#df_retrain_metrics_genconvit_speak.to_csv(os.path.join(output_dir, "retrain_metrics_genconvit_speak.csv"), index=False)

df_metrics_benchmark = pd.DataFrame(metrics_data_benchmark, columns=headers_bench)
#df_metrics_benchmark.to_csv(os.path.join(output_dir, "overall_benchmark_metrics.csv"), index=False)

df_metrics_benchmark_retrain = pd.DataFrame(metrics_data_benchmark_retrain, columns=headers_bench)
#df_metrics_benchmark_retrain.to_csv(os.path.join(output_dir, "overall_benchmark_metrics_retrain.csv"), index=False)

print("\n Original GENCONVIT metrics:\n")
print(df_original_metrics_genconvit)
print("\n Retrained GENCONVIT metrics in WildDeepfake:\n")
print(df_retrain_metrics_genconvit_wild)
print("\n Retrained GENCONVIT metrics in DeepSpeak:\n")
print(df_retrain_metrics_genconvit_speak)
print("\n Original DeepfakeBenchmark metrics on DeepSpeak:\n")
print(df_metrics_benchmark)
print("\n Retrained DeepfakeBenchmark metrics on DeepSpeak:\n")
print(df_metrics_benchmark_retrain)