# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:55:29 2021

@author: katrine
"""
import numpy as np


def get_seg_errors_mask(pred_labels, ref_labels):
    # pred_labels: [#slices, nclasses, x, y]
    # ref_labels have shape: [#slices, x, y]
    err_indices = np.zeros_like(ref_labels)

    for cls_idx in np.arange(0, pred_labels.shape[1]):
        pr_labels = pred_labels[:, cls_idx]
        err_indices[(pr_labels == 1) != (ref_labels == cls_idx)] = 1

    return err_indices.astype(np.bool)


def generate_thresholds(pred_labels, ref_labels, umap):
    err_indices = get_seg_errors_mask(pred_labels, ref_labels)
    umap_err = umap[err_indices]
    percentiles = [np.percentile(umap_err, p) for p in np.arange(1, 101)]
    percentiles = [0] + percentiles
    return percentiles