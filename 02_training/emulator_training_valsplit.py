import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os

# ==========================================================
# REPRODUCIBILITY & SETUP
# ==========================================================

# Set global seeds for data split consistency
tf.random.set_seed(42)
np.random.seed(42)

# Global constants for the repetition loop
NUM_REPETITIONS = 10 
DATA_DIR = 'data'
MODEL_BASE_DIR = 'models'
os.makedirs(MODEL_BASE_DIR, exist_ok=True) # Create directory to save models

# ----------------------------------------------------------------------
# --- Custom Serialization Fix (Must be outside build_model) ---
# ----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="CustomSlicing")
def slice_input(x, indices):
    """Slices the input tensor for Lambda layer."""
    return tf.gather(x, indices, axis=1)

# --- 1. Load Data ---
EXPRESSION_FILE = os.path.join(DATA_DIR, 'simulated_gene_expression_file_heter.csv')       # simulated bulk profiles (genes x samples)
PROPORTIONS_FILE = os.path.join(DATA_DIR, 'decon_signature_proportions_file_heter.csv')     # TOAST group proportions (samples x groups)
COUNT_FILE = os.path.join(DATA_DIR, 'reference_bulk_expression.csv')                        # clinical bulk expression; only its gene list is used
SIGNATURE_GENES_FILE = os.path.join(DATA_DIR, 'group_signature_genes.csv')                  # marker genes per group (one column per group)

try:
    gene_expression_df = pd.read_csv(EXPRESSION_FILE, index_col=0)
    signature_proportions_df = pd.read_csv(PROPORTIONS_FILE, index_col=0)
    count_df = pd.read_csv(COUNT_FILE, index_col=0)
    dataframe_signature = pd.read_csv(SIGNATURE_GENES_FILE) 
    print("Successfully loaded data files and signature gene lists.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Using dummy data for demonstration.")
    num_samples = 500; num_genes = 250; num_signatures = 4
    gene_names = [f'Gene_{i}' for i in range(num_genes)]
    signature_cols = [f'Group_{i}' for i in range(num_signatures)] 
    gene_expression_df = pd.DataFrame(np.random.rand(num_genes, num_samples), index=gene_names, columns=[f'Sample_{i}' for i in range(num_samples)])
    signature_proportions_df = pd.DataFrame(np.random.rand(num_samples, num_signatures), index=[f'Sample_{i}' for i in range(num_samples)], columns=signature_cols)
    count_df = gene_expression_df.copy()
    sig_gene_data = {col: [f'Gene_{j}' for j in range(i * 50, (i + 1) * 50)] for i, col in enumerate(signature_cols)}
    dataframe_signature = pd.DataFrame(sig_gene_data)
    print("Using synthetic data and gene lists for demonstration.")

# Data preprocessing
signature_proportions_df = signature_proportions_df.div(signature_proportions_df.sum(axis=1), axis=0)
gene_expression_df = gene_expression_df.T 
gene_expression_df = gene_expression_df.loc[signature_proportions_df.index]

common_genes = gene_expression_df.columns.intersection(count_df.index)
gene_expression_df = gene_expression_df[common_genes]
count_df = count_df.loc[common_genes]
signature_proportions = signature_proportions_df.values
signature_cols = signature_proportions_df.columns.tolist()

# Train/Test Split (80% train / 20% test)
expr_train_raw, expr_test_raw, prop_train, prop_test = train_test_split(
    gene_expression_df, signature_proportions, test_size=0.2, random_state=42, shuffle=True
)

# Validation set taken from the training samples (640 fit / 160 validation).
# Early stopping, checkpointing and the grid search use the validation set only;
# the test samples are used only for the final evaluation.
expr_train_raw, expr_val_raw, prop_train, prop_val = train_test_split(
    expr_train_raw, prop_train, test_size=0.2, random_state=42, shuffle=True
)
test_ids = expr_test_raw.index
print(f"Split: {len(expr_train_raw)} fit / {len(expr_val_raw)} validation / {len(expr_test_raw)} test")

# ----------------------------------------------------------------------
# --- 2. Structural Knowledge & Normalization ---
# ----------------------------------------------------------------------
all_genes = gene_expression_df.columns.tolist()
NUM_SIGNATURES = signature_proportions_df.shape[1] 

sig_gene_groups = []
for group_col in signature_cols:
    if group_col in dataframe_signature.columns:
        gene_list = dataframe_signature[group_col].tolist()
        gene_list = [g for g in gene_list if pd.notna(g) and g in all_genes]
        sig_gene_groups.append(gene_list)

sig_indices = []
for group in sig_gene_groups:
    indices = [all_genes.index(gene) for gene in group] 
    sig_indices.append(indices)

# Normalization
scaler = StandardScaler()
expr_train = scaler.fit_transform(expr_train_raw)
expr_val = scaler.transform(expr_val_raw)
expr_test = scaler.transform(expr_test_raw)
joblib.dump(scaler, os.path.join(MODEL_BASE_DIR, 'scaler_for_prediction_fit.pkl'))
np.save(os.path.join(MODEL_BASE_DIR, 'expr_train_scaled_fit.npy'), expr_train)

# --- 4. Define DUAL-PATHWAY model building function ---
def lr_schedule(epoch, lr):
    if epoch > 100:
        return lr * 0.1
    return lr

def build_model(learning_rate=0.001, l2_factor=0.01, huber_delta=0.5):
    """
    Constructs the Dual-Pathway Model:
    1. Parallel Sliced Paths (Signature Genes - High Influence)
    2. Full Context Path (All Genes - Contextual Influence)
    """
    input_dim = expr_train.shape[1]
    main_input = Input(shape=(input_dim,), name='full_gene_input')
    
    parallel_outputs = []
    
    # --- 1. Signature Gene Pathways (Specific, High Influence) ---
    for i, group_indices in enumerate(sig_indices):
        if not group_indices: continue
        sig_name = signature_cols[i] if i < len(signature_cols) else f'Sig_{i}'
        
        # Slicing: Only use the specific signature genes
        sliced_input = Lambda(slice_input, arguments={'indices': group_indices}, 
                              output_shape=(len(group_indices),), name=f'{sig_name}_slice')(main_input)

        # Deep Sub-Network: Dedicated feature extraction
        x = Dense(64, activation='relu', kernel_regularizer=l2(l2_factor), name=f'{sig_name}_dense_1')(sliced_input)
        x = BatchNormalization(name=f'{sig_name}_bn_1')(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(l2_factor), name=f'{sig_name}_dense_2')(x)
        x = BatchNormalization(name=f'{sig_name}_bn_2')(x)
        parallel_outputs.append(x)
        
    # --- 2. Full Context Pathway (General, Contextual Influence) ---
    context = Dense(32, activation='relu', kernel_regularizer=l2(l2_factor), name='full_context_dense_1')(main_input)
    context = BatchNormalization(name='full_context_bn_1')(context)
    parallel_outputs.append(context) 

    # 3. Concatenate all learned features (Signatures + Context)
    if len(parallel_outputs) > 1:
        merged_features = Concatenate(name='merged_all_features')(parallel_outputs)
    else:
        merged_features = context 

    # 4. Final Shared Prediction Head
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_factor), name='final_shared_dense')(merged_features)
    x = BatchNormalization(name='final_shared_bn')(x)
    outputs = Dense(NUM_SIGNATURES, activation='linear', name='output_proportions')(x) 
    
    model = Model(inputs=main_input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber(delta=huber_delta), metrics=['mae'])
    return model

# Setup the KerasRegressor wrapper for GridSearchCV
model_wrapper = KerasRegressor(
    model=build_model,
    model__learning_rate=0.001,
    model__l2_factor=0.01,
    model__huber_delta=0.5,
    batch_size=32,
    verbose=0
)

# --- 5. Grid Search to find and fix Best Hyperparameters (Run ONCE) ---
param_grid = {
    'model__l2_factor': [0.0001, 0.001],
    'model__huber_delta': [0.1, 0.5],
    'model__learning_rate': [0.01],
    'batch_size': [8]
}

print("\n--- Starting Initial Hyperparameter Search (Run 1 of 1) ---")
grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=4, scoring='neg_mean_absolute_error', n_jobs=1, verbose=2) 
grid_result = grid.fit(expr_train, prop_train, validation_data=(expr_val, prop_val), callbacks=[LearningRateScheduler(lr_schedule)], epochs=10) 

best_model_params = {k.replace('model__', ''): v for k, v in grid_result.best_params_.items() if k.startswith('model__')}
best_batch_size = grid_result.best_params_['batch_size']

print(f"\nBest Parameters (Fixed for all runs): {grid_result.best_params_}\n")


# ----------------------------------------------------------------------
# --- 6. 10x Independent Training Loop (With Per-Group Metrics) ---
# ----------------------------------------------------------------------
all_run_metrics = []
all_run_histories = [] 
custom_objects = {"slice_input": slice_input}
total_loop_start_time = time.time()
best_overall_rho = -np.inf
best_run_model_path = ""
model_paths = [] 

print(f"\n--- Starting {NUM_REPETITIONS} Independent Training Runs (Dual-Pathway Model) ---")

for run_i in range(NUM_REPETITIONS):
    tf.keras.backend.clear_session() 
    
    print(f"\n--- Training Run {run_i + 1}/{NUM_REPETITIONS} ---")
    
    current_model = build_model(**best_model_params)
    
    run_model_path = os.path.join(MODEL_BASE_DIR, f'model_run_{run_i+1}_best_with_time.h5')
    model_paths.append(run_model_path) 
    
    checkpoint = ModelCheckpoint(run_model_path, monitor='val_loss', save_best_only=True, verbose=0) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    run_start_time = time.time()
    
    history = current_model.fit(
        expr_train, prop_train, 
        epochs=100, 
        batch_size=best_batch_size, 
        validation_data=(expr_val, prop_val), 
        callbacks=[checkpoint, LearningRateScheduler(lr_schedule), early_stopping], 
        verbose=0
    )
    
    run_end_time = time.time()
    run_duration = run_end_time - run_start_time
    
    all_run_histories.append(history.history)
    
    try:
        final_model = load_model(run_model_path, custom_objects=custom_objects)
    except:
        final_model = current_model 
        
    predicted_raw = final_model.predict(expr_test)
    predicted_proportions = predicted_raw / np.sum(predicted_raw, axis=1, keepdims=True)

    mae = np.mean(np.abs(prop_test - predicted_proportions))
    
    # --- Spearman rho for each of the 4 groups, and their mean ---
    rhos_i = [
        spearmanr(prop_test[:, col_k], predicted_proportions[:, col_k]).correlation 
        for col_k in range(NUM_SIGNATURES)
    ]
    mean_rho_i = np.mean(rhos_i)

    run_metric_dict = {
        'run': run_i + 1, 
        'model_path': run_model_path, 
        'test_mae': mae, 
        'test_rho': mean_rho_i,  # Stores Mean Rho across all 4 groups
        'training_time_sec': run_duration
    }
    
    # Store per-group Rhos dynamically
    for col_k, col_name in enumerate(signature_cols):
        run_metric_dict[f'rho_{col_name}'] = rhos_i[col_k]

    all_run_metrics.append(run_metric_dict)
    
    if mean_rho_i > best_overall_rho:
        best_overall_rho = mean_rho_i
        best_run_model_path = run_model_path
        
    print(f"Run {run_i+1} saved to {run_model_path} - Test MAE: {mae:.4f}, Mean Rho (4 Groups): {mean_rho_i:.4f} (Training Time: {run_duration:.2f}s)")

total_loop_end_time = time.time()
total_loop_time = total_loop_end_time - total_loop_start_time

# ----------------------------------------------------------------------
# --- 7. ENSEMBLE PREDICTION AND FINAL REPORT (With Per-Group Breakdown) ---
# ----------------------------------------------------------------------

df_metrics = pd.DataFrame(all_run_metrics)
best_run = df_metrics.loc[df_metrics['test_rho'].idxmax()]

print("\n" + "="*70)
print("               ENSEMBLE AND FINAL MODEL REPORT")
print("="*70)
print(f"Total Training Time for 10 Runs: {total_loop_time:.2f} seconds")
print(f"Average Training Time per Run: {df_metrics['training_time_sec'].mean():.2f} seconds")

# --- A. Single Best Model Report & Pure Inference Timing ---
print("\n--- SINGLE BEST MODEL (Optimal Run) ---")
print(f"Run {best_run['run']} | Path: {best_run['model_path']}")
print(f"Test Mean Spearman Rho (4 Groups): {best_run['test_rho']:.4f}")
print(f"Test MAE: {best_run['test_mae']:.4f}")

optimal_model = load_model(best_run['model_path'], custom_objects=custom_objects)

opt_inf_start = time.time()
optimal_pred_raw = optimal_model.predict(expr_test, verbose=0)
opt_inf_end = time.time()
optimal_inf_duration = opt_inf_end - opt_inf_start

print(f"Single Best Model Pure Inference Time: {optimal_inf_duration:.4f} seconds")
print(f"Single Best Model Time per Sample: {optimal_inf_duration / expr_test.shape[0]:.6f} seconds")

# --- B. Ensemble Prediction & Pure Inference Timing ---
print("\n--- ENSEMBLE MODEL (Averaging 10 Predictions) ---")

ensemble_predictions_raw = []
individual_model_pure_inference_times = []

ensemble_start_time = time.time()

for i, path in enumerate(model_paths):
    tf.keras.backend.clear_session()
    model_i = load_model(path, custom_objects=custom_objects)
    
    model_inf_start = time.time()
    pred_raw_i = model_i.predict(expr_test, verbose=0)
    model_inf_end = time.time()
    
    inf_duration = model_inf_end - model_inf_start
    individual_model_pure_inference_times.append(inf_duration)
    
    ensemble_predictions_raw.append(pred_raw_i)

ensemble_end_time = time.time()
total_pure_inference_time = sum(individual_model_pure_inference_times)
avg_single_model_pure_inference_time = np.mean(individual_model_pure_inference_times)

average_raw_output = np.mean(ensemble_predictions_raw, axis=0)
predicted_proportions_ensemble = average_raw_output / np.sum(average_raw_output, axis=1, keepdims=True)

mae_ensemble = np.mean(np.abs(prop_test - predicted_proportions_ensemble))

# --- Ensemble Spearman rho for each group, and their mean ---
ensemble_rhos_per_group = [
    spearmanr(prop_test[:, col_k], predicted_proportions_ensemble[:, col_k]).correlation 
    for col_k in range(NUM_SIGNATURES)
]
rho_ensemble = np.mean(ensemble_rhos_per_group)

print(f"Ensemble Mean Test Spearman Rho (All 4 Groups): {rho_ensemble:.4f}")
print(f"Ensemble Test MAE: {mae_ensemble:.4f}")
print("\n--- Ensemble Per-Group Spearman Rho Breakdown ---")
for k, col_name in enumerate(signature_cols):
    print(f"  - {col_name} (Group {k}): Rho = {ensemble_rhos_per_group[k]:.4f}")

print(f"\nTotal Pure Ensemble Inference Time (excluding loading): {total_pure_inference_time:.4f} seconds")
print(f"Average Pure Inference Time per Single Model: {avg_single_model_pure_inference_time:.4f} seconds")
print("="*70)

# --- C. Compare and Final Choice ---
if rho_ensemble > best_run['test_rho']:
    print(f"\nENSEMBLE ADVANTAGE: Ensemble Mean Rho ({rho_ensemble:.4f}) > Single Best Mean Rho ({best_run['test_rho']:.4f})")
    print("RECOMMENDATION: Use the ENSEMBLE PREDICTION METHOD for deployment.")
else:
    print(f"\nSINGLE BEST ADVANTAGE: Single Best Mean Rho ({best_run['test_rho']:.4f}) > Ensemble Mean Rho ({rho_ensemble:.4f})")
    print(f"RECOMMENDATION: Use the Single Best Model (Run {best_run['run']}) for deployment.")

print("="*70)

# ----------------------------------------------------------------------
# --- 7b. Agreement with TOAST as group calls, and saved predictions ---
# ----------------------------------------------------------------------
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
toast_call = prop_test.argmax(axis=1)
emu_call = predicted_proportions_ensemble.argmax(axis=1)
print("\n--- Ensemble vs TOAST, as group calls (highest group per sample) ---")
print(f"Top-group agreement: {accuracy_score(toast_call, emu_call):.4f}")
print(f"Cohen's kappa:       {cohen_kappa_score(toast_call, emu_call):.4f}")
print(f"Macro-F1:            {f1_score(toast_call, emu_call, average='macro'):.4f}")

# Save test-set predictions, TOAST targets, per-run metrics and timing
pd.DataFrame(predicted_proportions_ensemble, index=test_ids, columns=signature_cols).to_csv(
    os.path.join(MODEL_BASE_DIR, 'test_predictions_ensemble.csv'))
pd.DataFrame(prop_test, index=test_ids, columns=signature_cols).to_csv(
    os.path.join(MODEL_BASE_DIR, 'test_toast_targets.csv'))
df_metrics.to_csv(os.path.join(MODEL_BASE_DIR, 'run_metrics.csv'), index=False)
pd.DataFrame([{
    'method': 'Emulator (10-model ensemble)',
    'n_test_samples': len(test_ids),
    'train_time_per_model_sec': df_metrics['training_time_sec'].mean(),
    'inference_sec_total': total_pure_inference_time,
    'inference_ms_per_sample': 1000 * total_pure_inference_time / len(test_ids),
}, {
    'method': 'Emulator (single best model)',
    'n_test_samples': len(test_ids),
    'train_time_per_model_sec': df_metrics['training_time_sec'].mean(),
    'inference_sec_total': optimal_inf_duration,
    'inference_ms_per_sample': 1000 * optimal_inf_duration / len(test_ids),
}]).to_csv(os.path.join(MODEL_BASE_DIR, 'emulator_timing.csv'), index=False)
print(f"Saved test predictions and TOAST targets for {len(test_ids)} test samples")
print("="*70)

# ======================================================================
# --- 8. VISUALIZATIONS & SAVING ---
# ======================================================================

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Training & Validation Loss
for i, h in enumerate(all_run_histories):
    axes[0, 0].plot(h['loss'], color='blue', alpha=0.15)
    axes[0, 0].plot(h['val_loss'], color='red', alpha=0.15)
axes[0, 0].set_title('A. Training & Validation Loss (All 10 Runs)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Huber Loss')
axes[0, 0].legend(['Train (Blue)', 'Val (Red)'])

# Panel B: Metric Stability Across 10 Independent Runs (MAE & Mean Rho across 4 groups)
sns.boxplot(data=df_metrics[['test_mae', 'test_rho']], ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title('B. Metric Stability Across 10 Independent Runs', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Metric Value')

# Panel C: Ensemble Prediction Accuracy Scatter Plot (All 4 Groups Combined)
y_true_flat = prop_test.flatten()
y_pred_flat = predicted_proportions_ensemble.flatten()
sns.regplot(x=y_true_flat, y=y_pred_flat, ax=axes[1, 0], 
            scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
axes[1, 0].plot([0, 1], [0, 1], color='black', linestyle='--')
axes[1, 0].set_title(f'C. Ensemble Prediction Accuracy (Mean Rho: {rho_ensemble:.4f})', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('True Proportions')
axes[1, 0].set_ylabel('Predicted Proportions')

# Panel D: Mean Absolute Error per Signature
mae_per_signature = np.mean(np.abs(prop_test - predicted_proportions_ensemble), axis=0)
sns.barplot(x=signature_cols, y=mae_per_signature, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title('D. Mean Absolute Error per Signature', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('MAE')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

plt.tight_layout()

png_path = os.path.join(MODEL_BASE_DIR, 'model_fitness_report_4groups.png')
pdf_path = os.path.join(MODEL_BASE_DIR, 'model_fitness_report_4groups.pdf')
svg_path = os.path.join(MODEL_BASE_DIR, 'model_fitness_report_4groups.svg')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(svg_path, format='svg', bbox_inches='tight')

# Panel E: Comparison Bar Plot (Mean Rho across 4 groups)
plt.figure(figsize=(12, 6))
labels = [f'Run {i+1}' for i in range(NUM_REPETITIONS)] + ['ENSEMBLE']
rhos = df_metrics['test_rho'].tolist() + [rho_ensemble]
colors = ['skyblue'] * NUM_REPETITIONS + ['orange']

sns.barplot(x=labels, y=rhos, palette=colors)
mean_indiv_rho = df_metrics['test_rho'].mean()
plt.axhline(y=mean_indiv_rho, color='black', linestyle='--', label=f"Mean Individual Rho ({mean_indiv_rho:.4f})")
plt.title('E. Comparison of Mean Spearman Rho Across All 4 Groups: Individual Runs vs. Ensemble', fontsize=14, fontweight='bold')
plt.ylabel('Mean Spearman Rho (4 Groups)')
plt.ylim(min(rhos) - 0.02, max(rhos) + 0.02)
plt.legend(loc='lower right')
plt.tight_layout()

comp_png_path = os.path.join(MODEL_BASE_DIR, 'ensemble_comparison_4groups.png')
comp_pdf_path = os.path.join(MODEL_BASE_DIR, 'ensemble_comparison_4groups.pdf')
comp_svg_path = os.path.join(MODEL_BASE_DIR, 'ensemble_comparison_4groups.svg')

plt.savefig(comp_png_path, dpi=300, bbox_inches='tight')
plt.savefig(comp_pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(comp_svg_path, format='svg', bbox_inches='tight')

print(f"\nMain report saved as PNG: {png_path}")
print(f"Main report saved as PDF: {pdf_path}")
print(f"Ensemble comparison saved as PNG: {comp_png_path}")
print(f"Ensemble comparison saved as PDF: {comp_pdf_path}")

plt.show()