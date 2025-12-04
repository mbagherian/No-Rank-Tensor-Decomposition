# no-rank-tensor-decomposition/metric_learning/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def visualize_tensor_comparison_with_reconstruction(methods, labels, results, dataset_name, original_data):
    """Comprehensive visualization with manifold plots, reconstruction metrics, and sample reconstructions"""
    fig = plt.figure(figsize=(28, 20))
    
    # Create grid layout
    gs = plt.GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.4)
    
    # Get all available methods and prioritize important ones
    all_methods = list(methods.keys())
    
    # Strategic method selection to ensure diversity
    priority_methods = []
    
    # 1. Always include Metric Learning first
    ml_key = 'Metric Learning ' if 'Metric Learning ' in all_methods else 'Metric Learning'
    if ml_key in all_methods:
        priority_methods.append(ml_key)
        all_methods.remove(ml_key)
    
    # 2. Include CP methods
    cp_methods = [m for m in all_methods if 'CP' in m]
    priority_methods.extend(cp_methods[:2])  # Take first 2 CP
    
    # 3. Include Tucker methods  
    tucker_methods = [m for m in all_methods if 'Tucker' in m]
    priority_methods.extend(tucker_methods[:2])  # Take first 2 Tucker
    
    # 4. Fill remaining slots with other methods
    remaining_slots = 6 - len(priority_methods)
    if remaining_slots > 0:
        other_methods = [m for m in all_methods if m not in priority_methods]
        priority_methods.extend(other_methods[:remaining_slots])
    
    print(f"📊 Plotting methods: {priority_methods}")
    
    # Plot 1-6: Manifold visualizations
    for i, method_name in enumerate(priority_methods[:6]):
        method_data = methods[method_name]
        if method_data is None:
            continue
            
        ax = fig.add_subplot(gs[i//3, i%3])
        embeddings = method_data['embeddings']
        
        # Reduce dimensionality for visualization if needed
        if embeddings.shape[1] > 2:
            viz_data = PCA(n_components=2).fit_transform(embeddings)
        else:
            viz_data = embeddings
            
        scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], c=labels,
                           cmap='viridis', alpha=0.7, s=50)
        
        # Find corresponding result for this method
        method_result = next((r for r in results if r['Method'] == method_name), None)
        if method_result:
            # Create compact title
            short_name = method_name.replace('Decomposition', '').replace('+ K-Means', '').strip()
            if method_name == 'Metric Learning ':
                short_name = 'Metric Learning'
            
            title = f'{short_name}\nSil: {method_result["Silhouette"]:.3f}'
            if method_result.get('Reconstruction_Error') is not None:
                title += f'\nErr: {method_result["Reconstruction_Error"]:.3f}'
            ax.set_title(short_name, fontsize=11, pad=15)
        
        ax.set_xlabel('PC1' if embeddings.shape[1] > 2 else 'Dim1', fontsize=9)
        ax.set_ylabel('PC2' if embeddings.shape[1] > 2 else 'Dim2', fontsize=9)
        ax.tick_params(labelsize=8)
    
    # Plot 7: Reconstruction metrics comparison
    ax_recon = fig.add_subplot(gs[2, :3])
    
    reconstruction_methods = ['CP-R5', 'CP-R10', 'CP-R20', 'Tucker-R5', 'Tucker-R10', 'Tucker-R20', 'Metric Learning']
    recon_errors = []
    exp_variances = []
    valid_methods = []
    
    for method_name in reconstruction_methods:
        plot_name = 'Metric Learning ' if method_name == 'Metric Learning' else method_name
        if plot_name in methods and methods[plot_name] is not None:
            if 'reconstruction_metrics' in methods[plot_name]:
                metrics = methods[plot_name]['reconstruction_metrics']
                recon_errors.append(metrics['reconstruction_error'])
                exp_variances.append(metrics['explained_variance'])
                valid_methods.append(method_name)
    
    if valid_methods:
        x = np.arange(len(valid_methods))
        width = 0.35
        
        bars1 = ax_recon.bar(x - width/2, recon_errors, width, label='Reconstruction Error', 
                           color='red', alpha=0.7)
        ax_recon.set_ylabel('Reconstruction Error', color='red', fontsize=11)
        ax_recon.tick_params(axis='y', labelcolor='red', labelsize=9)
        
        ax2 = ax_recon.twinx()
        bars2 = ax2.bar(x + width/2, exp_variances, width, label='Explained Variance', 
                       color='blue', alpha=0.7)
        ax2.set_ylabel('Explained Variance', color='blue', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=9)
        
        ax_recon.set_title('Reconstruction Performance', fontsize=12, pad=15)
        ax_recon.set_xticks(x)
        
        short_labels = [label.replace('Tucker-', 'T-').replace('CP-', 'C-') 
                       for label in valid_methods]
        ax_recon.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
        
        # Add value labels
        for bar, value in zip(bars1, recon_errors):
            ax_recon.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=8, color='red')
        
        for bar, value in zip(bars2, exp_variances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8, color='blue')
        
        ax_recon.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
    
    # Plot 8: Structural metrics comparison
    ax_metrics = fig.add_subplot(gs[2, 3:])
    metric_names = ['Separation_Ratio', 'Continuity', 'Trustworthiness', 'ARI', 'NMI']
    
    # Use the same methods as in scatter plots
    plot_methods_data = [(name, next((r for r in results if r['Method'] == name), None))
                        for name in priority_methods[:6]]
    plot_methods_data = [pm for pm in plot_methods_data if pm[1] is not None]
    
    if plot_methods_data:
        x = np.arange(len(metric_names))
        width = 0.13
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_methods_data)))
        
        for i, (method_name, result) in enumerate(plot_methods_data):
            method_metrics = [result[metric] for metric in metric_names]
            
            # Create shorter legend labels
            short_legend = method_name.replace('Decomposition', '').replace('+ K-Means', '').strip()
            if method_name == 'Metric Learning ':
                short_legend = 'Metric Learning'
            if len(short_legend) > 12:
                short_legend = short_legend[:10] + '...'
                
            ax_metrics.bar(x + i*width, method_metrics, width, label=short_legend, 
                          color=colors[i], alpha=0.7)
        
        ax_metrics.set_title('Structural Metrics', fontsize=12, pad=15)
        ax_metrics.set_xticks(x + width*(len(plot_methods_data)-1)/2)
        
        xlabels = ['Separation\nRatio', 'Continuity', 'Trust-\nworthiness', 'ARI', 'NMI']
        ax_metrics.set_xticklabels(xlabels, fontsize=10)
        ax_metrics.set_ylabel('Score', fontsize=11)
        ax_metrics.tick_params(axis='y', labelsize=9)
        
        ax_metrics.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Plot 9: Sample reconstructions
    ax_samples = fig.add_subplot(gs[3, :])
    ax_samples.axis('off')
    ax_samples.set_title(f'Sample Reconstructions - {dataset_name}', fontsize=14, pad=20)
    
    # Find methods that have reconstructions
    recon_methods = []
    for method_name, method_data in methods.items():
        if method_data is not None and 'reconstructed' in method_data:
            recon_methods.append((method_name, method_data['reconstructed']))
    
    if recon_methods:
        # Show sample reconstructions in a separate figure
        show_sample_reconstructions_separate(methods, original_data, labels, dataset_name)
    else:
        ax_samples.text(0.5, 0.5, 'No reconstruction data available', 
                       ha='center', va='center', transform=ax_samples.transAxes, fontsize=12)
    
    plt.suptitle(f'Comprehensive Tensor Analysis - {dataset_name}', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.show()

def show_sample_reconstructions_separate(methods, original_data, labels, dataset_name):
    """Show sample reconstructions in a separate figure"""
    # Find methods that have reconstructions
    recon_methods = []
    for method_name, method_data in methods.items():
        if method_data is not None and 'reconstructed' in method_data:
            recon_methods.append((method_name, method_data['reconstructed']))
    
    if not recon_methods:
        return
    
    # Get unique classes
    unique_labels = np.unique(labels)
    n_samples = min(4, len(unique_labels))
    n_methods = len(recon_methods) + 1  # +1 for original
    
    fig, axes = plt.subplots(n_samples, n_methods, figsize=(3*n_methods, 3*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Get one sample per class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = idx
        if len(class_indices) >= n_samples:
            break
    
    # Column headers
    axes[0, 0].set_title('Original', fontsize=12, pad=10)
    for j, (method_name, _) in enumerate(recon_methods, 1):
        short_name = method_name.replace('Tucker-', 'T-').replace('CP-', 'C-')
        if len(short_name) > 10:
            short_name = short_name[:8] + '...'
        axes[0, j].set_title(short_name, fontsize=10, pad=10)
    
    for i, (label, idx) in enumerate(class_indices.items()):
        if i >= n_samples:
            break
            
        # Original
        axes[i, 0].imshow(original_data[idx], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(f"Class {label}", fontsize=10, rotation=0, ha='right', va='center')
        
        # Reconstructions
        for j, (method_name, reconstructed) in enumerate(recon_methods, 1):
            axes[i, j].imshow(reconstructed[idx], cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle(f'Sample Reconstructions - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_tensor_decomposition_comparison(results_df, dataset_name):
    """Simple bar chart comparison of all metrics across methods"""
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.88, bottom=0.25)
    
    methods = results_df['Method'].values
    metrics_to_plot = [
        ('Silhouette', 'Silhouette Score', 'skyblue'),
        ('Davies-Bouldin', 'Davies-Bouldin Index', 'lightcoral'),
        ('Separation_Ratio', 'Separation Ratio', 'lightgreen'),
        ('Continuity', 'Continuity', 'violet'),
        ('Trustworthiness', 'Trustworthiness', 'orange'),
        ('ARI', 'Adjusted Rand Index', 'gold'),
        ('NMI', 'Normalized Mutual Info', 'lightpink')
    ]
    
    # Plot first 7 metrics
    for idx, (metric_col, metric_name, color) in enumerate(metrics_to_plot):
        if idx >= 7:
            break
        
        ax = axes[idx // 4, idx % 4]
        values = results_df[metric_col].values
        
        bars = ax.bar(methods, values, color=color, alpha=0.7, width=0.6)
        ax.set_title(metric_name, fontsize=12, pad=15)
        ax.set_ylabel('Score', fontsize=10)
        
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.set_xticklabels(methods, rotation=90, ha='center')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Reconstruction metrics plot
    ax_recon = axes[1, 3] if len(axes.flat) > 7 else axes[1, 2]
    
    recon_methods = results_df[results_df['Reconstruction_Error'].notna()]
    if not recon_methods.empty:
        methods_recon = recon_methods['Method'].values
        recon_errors = recon_methods['Reconstruction_Error'].values
        exp_vars = recon_methods['Explained_Variance'].values
        
        x = np.arange(len(methods_recon))
        width = 0.35
        
        bars1 = ax_recon.bar(x - width / 2, recon_errors, width, label='Recon Error', color='red', alpha=0.7)
        ax_recon.set_ylabel('Reconstruction Error', color='red', fontsize=10)
        ax_recon.tick_params(axis='y', labelcolor='red')
        
        ax2 = ax_recon.twinx()
        bars2 = ax2.bar(x + width / 2, exp_vars, width, label='Explained Var', color='blue', alpha=0.7)
        ax2.set_ylabel('Explained Variance', color='blue', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax_recon.set_title('Reconstruction Metrics', fontsize=12, pad=15)
        ax_recon.set_xticks(x)
        ax_recon.set_xticklabels(methods_recon, rotation=90, ha='center', fontsize=8)
    
    plt.suptitle(f'Tensor Decomposition Metrics - {dataset_name}', fontsize=18)
    plt.show()