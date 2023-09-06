import open3d as o3d

from metrics_3d.visualize import threshold_plot_curves, pr_plot_curves
from metrics_3d.precision_recall import PrecisionRecall

N_SAMPLE = 1000

# define metrics
pr_metric = PrecisionRecall(min_t=0.001, max_t=0.01, num=100)

precision = []
recall = []
fscore = []
legend = []

# creating some shape for demonstration purposes
gt = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.04)
gt = gt.subdivide_loop(number_of_iterations=4)
gt_pcd = gt.sample_points_uniformly(N_SAMPLE)

for idx, r in enumerate([0.05, 0.045, 0.0425, 0.055, 0.0525, 0.06, 0.0575]):

    # creating some shape for demonstration purposes
    pt = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=50)
    pt_pcd = pt.sample_points_uniformly(N_SAMPLE)

    # the update method accumulates metrics
    pr_metric.update(gt, pt)  # mesh mesh
    pr_metric.update(gt_pcd, pt)  # pcd mesh
    pr_metric.update(gt, pt_pcd)  # mesh pcd
    pr_metric.update(gt_pcd, pt_pcd)  # pcd pcd

    # precision recall metric has different way to get a single number
    # pr, re, f1, t = pr_metric.compute_at_threshold(0.005)
    # pr, re, f1 = pr_metric.compute_auc()
    pr, re, f1 = pr_metric.compute_at_all_thresholds()
    precision.append(pr)
    recall.append(re)
    fscore.append(f1)
    legend.append('method {}'.format(idx + 1))

    # this reset all important variables in the pr class
    pr_metric.reset()

threshold_plot_curves(precision,
                      pr_metric.thresholds,
                      ylabel='precision [%]',
                      title='Precision Scores at Different Thresholds',
                      legend=legend)

threshold_plot_curves(recall,
                      pr_metric.thresholds,
                      ylabel='recall [%]',
                      title='Recall Scores at Different Thresholds',
                      legend=legend)

threshold_plot_curves(fscore,
                      pr_metric.thresholds,
                      ylabel='fscore [%]',
                      title='F-score at Different Thresholds',
                      legend=legend)

pr_plot_curves(precision,
               recall,
               ylabel='precision [%]',
               xlabel='recall [%]',
               title='Precision Recall Curves',
               legend=legend)
