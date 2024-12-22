import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_lgb_training_process(cv_results, figsize=(12, 6)):
    """
    Plot LightGBM training process showing validation AUC scores over iterations
    
    Parameters:
    -----------
    cv_results : list
        List of validation AUC scores for each fold during training
    figsize : tuple
        Figure size for the plot
    """
    # plt.style.use('seaborn')
    plt.figure(figsize=figsize)
    
    # Plot each fold
    colors = sns.color_palette("husl", len(cv_results))
    for fold, (scores, color) in enumerate(zip(cv_results, colors), 1):
        iterations = range(200, (len(scores) + 1) * 200, 200)
        plt.plot(iterations, scores, label=f'Fold {fold}', 
                color=color, alpha=0.8, linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Validation AUC', fontsize=12)
    plt.title('LightGBM Training Process', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Set y-axis limits with some padding
    plt.ylim(min(min(scores) for scores in cv_results) - 0.005,
             max(max(scores) for scores in cv_results) + 0.005)
    
    plt.tight_layout()
    return plt.gcf()

def plot_cv_results(cv_scores, figsize=(12, 6)):
    """
    Plot final cross-validation results
    
    Parameters:
    -----------
    cv_scores : list
        Final CV AUC scores for each fold
    figsize : tuple
        Figure size for the plot
    """
    # plt.style.use('seaborn')
    plt.figure(figsize=figsize)
    
    folds = range(1, len(cv_scores) + 1)
    mean_score = np.mean(cv_scores)
    
    # Create bar plot
    colors = sns.color_palette("husl", len(cv_scores))
    bars = plt.bar(folds, cv_scores, alpha=0.8, color=colors)
    plt.axhline(y=mean_score, color='r', linestyle='--', 
                label=f'Mean AUC: {mean_score:.4f}')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('LightGBM Cross-Validation Results', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set axis limits and ticks
    plt.ylim(min(cv_scores) - 0.002, max(cv_scores) + 0.002)
    plt.xticks(folds)
    
    plt.tight_layout()
    return plt.gcf()

# Example usage:
# Prepare data
cv_results = [
    [0.817627, 0.833834, 0.840857, 0.844994, 0.847601], # Fold 1
    [0.817604, 0.833714, 0.840934, 0.844969, 0.847588], # Fold 2
    [0.822135, 0.838778, 0.845986, 0.850200, 0.852911], # Fold 3
    [0.822161, 0.838682, 0.845788, 0.849880, 0.852535], # Fold 4
    [0.815578, 0.833114, 0.840380, 0.844591, 0.847380]  # Fold 5
]

cv_scores = [0.8580563718101737, 0.8577294797856132, 0.8629975559172397, 
             0.8631073423011661, 0.8583519651293828]

# Create plots
fig1 = plot_lgb_training_process(cv_results)
plt.figure(1)
plt.show()

fig2 = plot_cv_results(cv_scores)
plt.figure(2)
plt.show()