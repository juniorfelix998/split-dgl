"""
Comparison Script: Original vs Improved DGL Split Learning
===========================================================
This script compares the original and improved implementations
"""

import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
from tabulate import tabulate
import seaborn as sns

sns.set_style("whitegrid")


def load_all_results():
    """Load results from all implementations"""
    results = {}

    try:
        results['original'] = torch.load('dgl_split_learning_results.pt')
        print("✓ Loaded Original DGL results")
    except:
        print("✗ Original DGL results not found")
        results['original'] = None

    try:
        results['improved'] = torch.load('improved_dgl_split_learning_results.pt')
        print("✓ Loaded Improved DGL results")
    except:
        print("✗ Improved DGL results not found")
        results['improved'] = None

    try:
        results['baseline'] = torch.load('baseline_split_learning_results.pt')
        print("✓ Loaded Baseline Split Learning results")
    except:
        print("✗ Baseline results not found")
        results['baseline'] = None

    return results


def create_three_way_comparison(results):
    """Create comprehensive three-way comparison"""

    fig = plt.figure(figsize=(20, 12))

    # Extract data
    methods = []
    data = {}

    if results['original']:
        methods.append('Original DGL')
        data['Original DGL'] = results['original']

    if results['improved']:
        methods.append('Improved DGL')
        data['Improved DGL'] = results['improved']

    if results['baseline']:
        methods.append('Standard SL')
        data['Standard SL'] = results['baseline']

    colors = {'Original DGL': 'steelblue', 'Improved DGL': 'green', 'Standard SL': 'coral'}

    # ========================================================================
    # PLOT 1: Test Accuracy Comparison
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    for method in methods:
        rounds = list(range(1, len(data[method]['test_acc']) + 1))
        ax1.plot(rounds, data[method]['test_acc'],
                 label=method, linewidth=2.5, color=colors[method])

    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # PLOT 2: Test Loss Comparison
    # ========================================================================
    ax2 = plt.subplot(2, 3, 2)
    for method in methods:
        rounds = list(range(1, len(data[method]['test_loss']) + 1))
        ax2.plot(rounds, data[method]['test_loss'],
                 label=method, linewidth=2.5, color=colors[method])

    ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # PLOT 3: Training Accuracy Comparison
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)
    for method in methods:
        rounds = list(range(1, len(data[method]['train_acc']) + 1))
        ax3.plot(rounds, data[method]['train_acc'],
                 label=method, linewidth=2.5, color=colors[method])

    ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # PLOT 4: Final Metrics Bar Chart
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)

    final_test_accs = [data[m]['test_acc'][-1] for m in methods]
    x = np.arange(len(methods))

    bars = ax4.bar(x, final_test_accs,
                   color=[colors[m] for m in methods],
                   alpha=0.8, edgecolor='black', linewidth=2)

    ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, final_test_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    # ========================================================================
    # PLOT 5: Overfitting Analysis (Train-Test Gap)
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)

    for method in methods:
        train_acc = np.array(data[method]['train_acc'])
        test_acc = np.array(data[method]['test_acc'])
        gap = train_acc - test_acc
        rounds = list(range(1, len(gap) + 1))
        ax5.plot(rounds, gap, label=method, linewidth=2, color=colors[method])

    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Train-Test Gap (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Overfitting Analysis (Lower is Better)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # PLOT 6: Improvement Summary
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate improvements
    if results['original'] and results['improved']:
        orig_acc = data['Original DGL']['test_acc'][-1]
        impr_acc = data['Improved DGL']['test_acc'][-1]
        improvement = impr_acc - orig_acc

        summary_text = f"""
IMPROVEMENT SUMMARY

Original DGL:
  Test Accuracy: {orig_acc:.2f}%

Improved DGL:
  Test Accuracy: {impr_acc:.2f}%

Improvement: {improvement:+.2f}%

Applied Enhancements:
  ✅ Dropout regularization
  ✅ Label smoothing
  ✅ Separate LR for auxiliary
  ✅ Reduced weight decay
  ✅ Increased local epochs
  ✅ Strong augmentation

Target: 88-91% test accuracy
Status: {'ACHIEVED ✅' if impr_acc >= 88 else 'IN PROGRESS'}
"""
    else:
        summary_text = "Run both implementations to see comparison"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('original_vs_improved_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved to: original_vs_improved_comparison.png")


def print_detailed_metrics(results):
    """Print detailed comparison table"""

    print("\n" + "=" * 80)
    print("DETAILED METRICS COMPARISON")
    print("=" * 80)

    table_data = []
    headers = ["Metric", "Original DGL", "Improved DGL", "Standard SL", "Best"]

    metrics = {
        'original': results['original'],
        'improved': results['improved'],
        'baseline': results['baseline']
    }

    # Filter out None results
    available = {k: v for k, v in metrics.items() if v is not None}

    if len(available) < 2:
        print("Not enough results to compare. Please run the experiments.")
        return

    # Final test accuracy
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name]:
            values.append(f"{metrics[name]['test_acc'][-1]:.2f}%")
        else:
            values.append("N/A")
    best_idx = np.argmax([metrics[n]['test_acc'][-1] if metrics[n] else 0
                          for n in ['original', 'improved', 'baseline']])
    table_data.append(["Final Test Accuracy", values[0], values[1], values[2],
                       ["Original", "Improved", "Standard"][best_idx]])

    # Peak test accuracy
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name]:
            values.append(f"{max(metrics[name]['test_acc']):.2f}%")
        else:
            values.append("N/A")
    table_data.append(["Peak Test Accuracy", values[0], values[1], values[2], "-"])

    # Final train accuracy
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name]:
            values.append(f"{metrics[name]['train_acc'][-1]:.2f}%")
        else:
            values.append("N/A")
    table_data.append(["Final Train Accuracy", values[0], values[1], values[2], "-"])

    # Overfitting gap
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name]:
            gap = metrics[name]['train_acc'][-1] - metrics[name]['test_acc'][-1]
            values.append(f"{gap:.2f}%")
        else:
            values.append("N/A")
    best_idx = np.argmin([abs(metrics[n]['train_acc'][-1] - metrics[n]['test_acc'][-1])
                          if metrics[n] else 999
                          for n in ['original', 'improved', 'baseline']])
    table_data.append(["Overfitting Gap", values[0], values[1], values[2],
                       ["Original", "Improved", "Standard"][best_idx]])

    # Final test loss
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name]:
            values.append(f"{metrics[name]['test_loss'][-1]:.4f}")
        else:
            values.append("N/A")
    best_idx = np.argmin([metrics[n]['test_loss'][-1] if metrics[n] else 999
                          for n in ['original', 'improved', 'baseline']])
    table_data.append(["Final Test Loss", values[0], values[1], values[2],
                       ["Original", "Improved", "Standard"][best_idx]])

    # Training time
    values = []
    for name in ['original', 'improved', 'baseline']:
        if metrics[name] and 'training_time' in metrics[name]:
            values.append(f"{metrics[name]['training_time'] / 60:.2f}m")
        else:
            values.append("N/A")
    table_data.append(["Training Time", values[0], values[1], values[2], "-"])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Calculate improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    if metrics['original'] and metrics['improved']:
        orig_test = metrics['original']['test_acc'][-1]
        impr_test = metrics['improved']['test_acc'][-1]
        improvement = impr_test - orig_test

        print(f"\nOriginal DGL → Improved DGL:")
        print(f"  Test Accuracy: {orig_test:.2f}% → {impr_test:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")

        if improvement > 0:
            print(f"  Status: ✅ IMPROVED by {improvement:.2f}%")
        else:
            print(f"  Status: ⚠️ Need more tuning")

        # Gap analysis
        orig_gap = metrics['original']['train_acc'][-1] - metrics['original']['test_acc'][-1]
        impr_gap = metrics['improved']['train_acc'][-1] - metrics['improved']['test_acc'][-1]
        gap_improvement = orig_gap - impr_gap

        print(f"\n  Overfitting Gap: {orig_gap:.2f}% → {impr_gap:.2f}%")
        print(f"  Gap Reduction: {gap_improvement:+.2f}%")

        if gap_improvement > 0:
            print(f"  Status: ✅ Less overfitting")
        else:
            print(f"  Status: ⚠️ May need more regularization")

    if metrics['improved'] and metrics['baseline']:
        impr_test = metrics['improved']['test_acc'][-1]
        base_test = metrics['baseline']['test_acc'][-1]
        diff = impr_test - base_test

        print(f"\nImproved DGL vs Standard Split Learning:")
        print(f"  Test Accuracy: {impr_test:.2f}% vs {base_test:.2f}%")
        print(f"  Difference: {diff:+.2f}%")

        if abs(diff) <= 2:
            print(f"  Status: ✅ COMPARABLE accuracy (within 2%)")
            print(f"  Communication: 50% reduction with DGL")
            print(f"  Verdict: ✅ DGL is competitive!")
        elif diff > 2:
            print(f"  Status: ✅ DGL is BETTER!")
        else:
            print(f"  Status: ⚠️ Need more improvements")

    print("=" * 80)


def main():
    """Main comparison function"""

    print("\n" + "=" * 80)
    print("ORIGINAL VS IMPROVED DGL SPLIT LEARNING COMPARISON")
    print("=" * 80)

    # Load results
    print("\n[1/3] Loading results...")
    results = load_all_results()

    # Create comparison plots
    print("\n[2/3] Creating comparison plots...")
    create_three_way_comparison(results)

    # Print detailed metrics
    print("\n[3/3] Generating detailed comparison...")
    print_detailed_metrics(results)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. original_vs_improved_comparison.png")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess

        subprocess.check_call(['pip', 'install', 'tabulate', '--break-system-packages'])
        from tabulate import tabulate

    main()