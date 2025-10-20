#-----This file defines the functions for evaluating model efficiency and accuracy --------#
#%% Import libraries
import time
import os
import joblib
import psutil
import threading
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

#%%## Define functions for 5-fold CV evaluation - BOTH Effciency and Accuracy *****************************
# === Efficiency metrics function ===
def get_efficiency_metrics(model, X_train, y_train, X_test, model_path='temp_model.joblib'):
    pid = os.getpid()
    process = psutil.Process(pid)
    peak_memory = [0]
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            try:
                mem = process.memory_info().rss / (1024 ** 2)  # MB
                peak_memory[0] = max(peak_memory[0], mem)
                time.sleep(0.05)
            except:
                break

    mem_thread = threading.Thread(target=monitor)
    mem_thread.start()

    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    stop_event.set()
    mem_thread.join()

    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    joblib.dump(model, model_path)
    model_size_MB = os.path.getsize(model_path) / (1024 ** 2)

    return {
        "train_time_sec": round(end_train - start_train, 3),
        "predict_time_sec": round(end_pred - start_pred, 3),
        "peak_memory_MB": round(peak_memory[0], 2),
        "model_size_MB": round(model_size_MB, 3),
        "y_pred": y_pred
    }

# === Evaluate fold metrics ===
def evaluate_fold(pipe, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    metrics = {
        'AUROC': roc_auc_score(y_val, y_prob),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'F1': f1_score(y_val, y_pred, zero_division=0),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    return metrics

# === Full CV + Test evaluation ===
def run_cv_and_test(X_train, y_train, X_test, y_test, pipe, model, sampler,
                    dataset_name, fold_metrics_csv='fold_level_metrics_pq5andpq9.csv',
                    efficiency_csv='efficiency_metrics_pq5andpq9.csv', plot_dir='plots'):

    if sampler == "No-sample":
         sampler_name = "No-sample"
    else:
         sampler_name = sampler.__class__.__name__

    model_id = f"{model.__class__.__name__}_{sampler_name}_{dataset_name}"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=93196)
    fold_metrics_list = []
    y_vals = []
    y_probs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        metrics = evaluate_fold(pipe, X_tr, y_tr, X_val, y_val)
        metrics.update({
            'model_id': model_id,
            'fold': fold + 1,
            'dataset': dataset_name,
            'imbalance_handling_technique': sampler_name
        })
        fold_metrics_list.append(metrics)

        # Store for ROC plotting
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_vals.append(y_val)
        y_probs.append(y_prob)

    # Save fold metrics CSV
    df_fold = pd.DataFrame(fold_metrics_list)
    if os.path.exists(fold_metrics_csv):
        df_existing = pd.read_csv(fold_metrics_csv)
        df_combined = pd.concat([df_existing, df_fold], ignore_index=True)
    else:
        df_combined = df_fold
    df_combined.to_csv(fold_metrics_csv, index=False)
    print(f"Saved fold metrics to {fold_metrics_csv}")

    # Plot ROC curves for folds
    plot_roc_curve_for_folds(y_vals, y_probs, model_id, plot_dir)

    # Train final model on full train set for test evaluation & efficiency
    efficiency = get_efficiency_metrics(pipe, X_train, y_train, X_test)
    y_pred_test = efficiency.pop("y_pred")
    y_prob_test = pipe.predict_proba(X_test)[:, 1]

    # Calculate performance on test set
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    performance_test = {
        "AUROC_test": roc_auc_score(y_test, y_prob_test),
        "Precision_test": precision_score(y_test, y_pred_test, zero_division=0),
        "Recall_test": recall_score(y_test, y_pred_test, zero_division=0),
        "F1_test": f1_score(y_test, y_pred_test, zero_division=0),
        "TP_test": tp,
        "TN_test": tn,
        "FP_test": fp,
        "FN_test": fn
    }

    # Save efficiency and test performance
    combined_metrics = {**efficiency, **performance_test,
                        "model_id": model_id,
                        "dataset": dataset_name,
                        "imbalance_handling_technique": sampler_name}

    df_eff = pd.DataFrame([combined_metrics])
    if os.path.exists(efficiency_csv):
        df_existing = pd.read_csv(efficiency_csv)
        df_combined = pd.concat([df_existing, df_eff], ignore_index=True)
    else:
        df_combined = df_eff
    df_combined.to_csv(efficiency_csv, index=False)
    print(f"Saved efficiency and test metrics to {efficiency_csv}")



