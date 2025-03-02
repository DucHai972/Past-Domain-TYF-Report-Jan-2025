import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from data.split_set import split_and_normalize_fullmap, save_dataset_to_csv
from models.resnet import Resnet

def add_score_column(model, device, test_dataset, timestep, strict):
    model.eval()
    dataloader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        pin_memory=torch.cuda.is_available(), 
        num_workers=16
    )
    
    scores = []
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Processing files"):
            data = data.to(device)
            outputs = model(data)
            scores.extend(torch.sigmoid(outputs).cpu().numpy().squeeze().tolist())
    
    test_dataset.data['Score'] = scores
    csv_name = f'scores_{timestep}_{"strict" if strict else "nostrict"}.csv'
    csv_score = f'/N/slate/tnn3/HaiND/01-06_report/result_fullmap/csv_score/{csv_name}'
    test_dataset.data.to_csv(csv_score, index=False)
    return csv_score

def compute_metrics(df):
    # Compute confusion categories using a 0.5 threshold
    false_pos = df[(df['Label'] == 0) & (df['Score'] >= 0.5)]
    true_pos = df[(df['Label'] == 1) & (df['Score'] >= 0.5)]
    false_neg = df[(df['Label'] == 1) & (df['Score'] < 0.5)]
    true_neg = df[(df['Label'] == 0) & (df['Score'] < 0.5)]
    
    label1 = {}
    label1['precision'] = len(true_pos) / (len(true_pos) + len(false_pos)) if (len(true_pos) + len(false_pos)) > 0 else 0
    label1['recall'] = len(true_pos) / (len(true_pos) + len(false_neg)) if (len(true_pos) + len(false_neg)) > 0 else 0
    label1['f1'] = 2 * (label1['precision'] * label1['recall']) / (label1['precision'] + label1['recall']) if (label1['precision'] + label1['recall']) > 0 else 0

    label0 = {}
    label0['precision'] = len(true_neg) / (len(true_neg) + len(false_neg)) if (len(true_neg) + len(false_neg)) > 0 else 0
    label0['recall'] = len(true_neg) / (len(true_neg) + len(false_pos)) if (len(true_neg) + len(false_pos)) > 0 else 0
    label0['f1'] = 2 * (label0['precision'] * label0['recall']) / (label0['precision'] + label0['recall']) if (label0['precision'] + label0['recall']) > 0 else 0

    metrics = {
        'file': df.attrs.get('source_file', 'unknown'),
        'false_positives': len(false_pos),
        'true_positives': len(true_pos),
        'false_negatives': len(false_neg),
        'true_negatives': len(true_neg),
        'label1': label1,
        'label0': label0,
        'confusion_matrix': [
            [len(true_neg), len(false_pos)],
            [len(false_neg), len(true_pos)]
        ]
    }
    return metrics

def save_metrics(metrics, filepath):
    output = (
        f"File: {metrics['file']}\n"
        f"False Positives: {metrics['false_positives']}\n"
        f"True Positives: {metrics['true_positives']}\n"
        f"False Negatives: {metrics['false_negatives']}\n"
        f"True Negatives: {metrics['true_negatives']}\n\n"
        f"Label 1:\n"
        f"Precision: {metrics['label1']['precision']:.2f}\n"
        f"Recall: {metrics['label1']['recall']:.2f}\n"
        f"F1 Score: {metrics['label1']['f1']:.2f}\n\n"
        f"Label 0:\n"
        f"Precision: {metrics['label0']['precision']:.2f}\n"
        f"Recall: {metrics['label0']['recall']:.2f}\n"
        f"F1 Score: {metrics['label0']['f1']:.2f}\n"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(output)
    print("Metrics have been saved to", filepath)

def plot_confusion_matrix(cm, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print("Confusion matrix saved to", filename)
    plt.close()

def plot_distribution(df, filename):
    # Ensure we extract the month from Filename in a robust way
    month_series = pd.to_datetime(
        df['Filename'].str.extract(r'_(\d{4})(\d{2})\d{2}_')[0] +
        df['Filename'].str.extract(r'_(\d{4})(\d{2})\d{2}_')[1],
        format='%Y%m', errors='coerce'
    ).dt.month
    df['Month'] = month_series
    filtered = df[df['Score'] >= 0.5]
    counts = filtered.groupby('Month').size()
    
    if counts.empty:
        print("No data available for distribution plot.")
        return

    counts.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Positive predictions')
    plt.title('Number of positive predictions per Month')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    print("Distribution plot saved to", filename)
    plt.close()

def main():
    # Running example: python eval_fullyear.py --timestep t2_rus4_cw1 && \
    
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--timestep', type=str, required=True, help='Time step')
    parser.add_argument('--strict', action='store_true', help='Strict evaluation')
    parser.add_argument('--fullmonth', action='store_true', help='Full month evaluation')
    args = parser.parse_args()

    timestep = args.timestep
    strict = args.strict
    fullmonth = args.fullmonth

    print("Time step:", timestep)
    print("Strict evaluation:", strict)
    print("Full month evaluation:", fullmonth)

    # Determine results directory based on fullmonth flag
    if fullmonth:
        RESULT_DIR = '/N/slate/tnn3/HaiND/01-06_report/result_fullmap/all_months'
    else:
        RESULT_DIR = '/N/slate/tnn3/HaiND/01-06_report/result_fullmap/storm_months'

    pos_ind = int(timestep.split('_')[0][1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Resnet(inp_channels=228,
                   num_residual_block=[2, 2, 2, 2],
                   num_class=1).to(device)
    model_path = f'/N/slate/tnn3/HaiND/01-06_report/result/model/trained_model_{timestep}.pth'

    if not os.path.exists(model_path):
        model_path = f'/N/slate/tnn3/HaiND/01-06_report/result/model/trained_model_{timestep}_2.pth'
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create test set
    _, _, test_dataset = split_and_normalize_fullmap(
        csv_file='/N/slate/tnn3/HaiND/01-06_report/csv/merra_full_new_2.csv',
        pos_ind=pos_ind,
        norm_type='new',
        small_set=False,
        under_sample=False,
        strict=strict
    )

    csv_path = '/N/slate/tnn3/HaiND/01-06_report/sandbox/test_full_label.csv'
    save_dataset_to_csv(test_dataset, csv_path)

    # Save predictions to CSV and evaluate
    score_csv = add_score_column(model=model, device=device, test_dataset=test_dataset,
                                 timestep=timestep, strict=strict)
    df = pd.read_csv(score_csv)
    df.attrs['source_file'] = score_csv  # Add file info to df for metrics

    # If not fullmonth, filter out months outside May-November using a regex extraction
    if not fullmonth:
        month_extracted = df['Filename'].str.extract(r'_(\d{4})(\d{2})')[1]
        df['Month'] = pd.to_numeric(month_extracted, errors='coerce')
        df = df[(df['Month'] >= 5) & (df['Month'] <= 11)]

    metrics = compute_metrics(df)
    # Print metric results
    print(f"File: {metrics['file']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print("Label 1 metrics:")
    print(f"  Precision: {metrics['label1']['precision']:.2f}")
    print(f"  Recall: {metrics['label1']['recall']:.2f}")
    print(f"  F1 Score: {metrics['label1']['f1']:.2f}")
    print("Label 0 metrics:")
    print(f"  Precision: {metrics['label0']['precision']:.2f}")
    print(f"  Recall: {metrics['label0']['recall']:.2f}")
    print(f"  F1 Score: {metrics['label0']['f1']:.2f}")

    # Save metrics output to file
    metrics_file = f'{RESULT_DIR}/metrics/{timestep}_{"strict" if strict else "nostrict"}.txt'
    save_metrics(metrics, metrics_file)

    # Plot and save confusion matrix
    cm_file = f'{RESULT_DIR}/confusion_matrix/{timestep}_{"strict" if strict else "nostrict"}.png'
    plot_confusion_matrix(metrics['confusion_matrix'], cm_file)

    # Plot and save distribution of positive predictions per month
    distribution_file = f'{RESULT_DIR}/distribution/{timestep}_{"strict" if strict else "nostrict"}.png'
    plot_distribution(df, distribution_file)

if __name__ == '__main__':
    main()

    file_list = [
    't10_norus_cw1',
    't10_rus10_cw1',
    't10_rus20_cw1',
    't10_rus30_cw1',
    't10_rus4_cw1',
    't12_norus_cw1',
    't12_rus10_cw1',
    't12_rus20_cw1',
    't12_rus30_cw1',
    't12_rus4_cw1',
    't14_norus_cw1',
    't14_rus10_cw1',
    't14_rus20_cw1',
    't14_rus30_cw1',
    't14_rus4_cw1',
    't16_norus_cw1',
    't16_rus10_cw1',
    't16_rus20_cw1',
    't16_rus30_cw1',
    't16_rus4_cw1',
    't18_norus_cw1',
    't18_rus10_cw1',
    't18_rus20_cw1',
    't18_rus30_cw1',
    't18_rus4_cw1',
    't2_norus_cw1',
    't2_rus10_cw1',
    't2_rus20_cw1',
    't2_rus30_cw1',
    't2_rus4_cw1',
    't4_norus_cw1',
    't4_rus10_cw1',
    't4_rus20_cw1',
    't4_rus30_cw1',
    't4_rus4_cw1',
    't6_norus_cw1',
    't6_rus10_cw1',
    't6_rus20_cw1',
    't6_rus30_cw1',
    't6_rus4_cw1',
    't8_norus_cw1',
    't8_rus10_cw1',
    't8_rus20_cw1',
    't8_rus30_cw1',
    't8_rus4_cw1'
]