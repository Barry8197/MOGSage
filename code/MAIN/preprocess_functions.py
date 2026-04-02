import networkx as nx
import astropy.stats
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from palettable import wesanderson
import sys
sys.inset.path(0 , './../MAIN/')
from network_functions import abs_bicorr, pearson_corr
import warnings

warnings.filterwarnings('ignore')

def DESEQ(expr , meta , condition, fit_type='parametric') : 
    """
    Conducts differential expression analysis using DESeq2 algorithm.

    Parameters:
        count_mtx (pandas.DataFrame): Count data for different genes.
        datMeta (pandas.DataFrame): Metadata for the samples in count_mtx.
        condition (str): Column in datMeta to use for condition separation.
        fit_type (str, optional): Statistical fitting type for VST transformation.

    Returns:
        numpy.ndarray: Variance Stabilized Transformed counts.
    """    

    inference = DefaultInference(n_cpus=8)
        
    dds = DeseqDataSet(
        counts=expr,
        metadata=meta,
        design_factors=condition,
        refit_cooks=True,
        inference=inference,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
        )
    
    dds.deseq2()

    DeseqDataSet.vst(dds_full , fit_type = fit_type)
    vsd = dds_full.layers["vst_counts"]
    
    return vsd

def data_preprocess(expr , meta , transcriptomics = False) :
    """
    Processes count matrix data by removing genes with zero expression across all samples.
    Optionally filters genes based on expression levels and calculates similarity matrices.

    Parameters:
        count_mtx (pd.DataFrame): A DataFrame containing the gene count data.
        datMeta (pd.Series or pd.DataFrame): Metadata associated with the samples in count_mtx.
        transcriptomics (bool): If true, performs additional gene filtering and similarity matrix calculations.

    Returns:
        pd.DataFrame: The processed count matrix.
        pd.Series or pd.DataFrame: The corresponding processed metadata.
    """    
    n_genes = expr.shape[1]
    expr = expr.loc[: , (expr != 0).any(axis=0)] # remove any genes with all 0 expression
    
    if gene_exp == True : 
        filtered_genes = filter_genes(expr.T.to_numpy(), design=None, group=datMeta, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7)

        # Example printing the filtered rows
        print("Keeping %i genes" % sum(filtered_genes))
        print("Removed %i genes" % (n_genes - sum(filtered_genes)))

        expr = expr.loc[: , filtered_genes]

        adjacency_matrix  = abs_bicorr(expr.T , mat_means=False)
    else : 
        adjacency_matrix  = pearson_corr(expr.T , mat_means=False)

    ku = adjacency_matrix.sum(axis= 1)
    zku = (ku-np.mean(ku))/np.sqrt(np.var(ku))

    to_keep = zku > -2

    print("Keeping %i Samples" % sum(to_keep))
    print("Removed %i Samples" % (len(to_keep) - sum(to_keep)))

    expr = expr.loc[to_keep]
    meta = meta.loc[to_keep]
    
    return expr , meta

def custom_cpm(counts, lib_size):
    """
    Computes Counts Per Million (CPM) normalization on count data.

    Parameters:
        counts (np.array): An array of raw gene counts.
        lib_size (float or np.array): The total counts in each library (sample).

    Returns:
        np.array: Normalized counts expressed as counts per million.
    """    
    return counts / lib_size * 1e6

def filter_genes(y, design=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """
    Filters genes based on several criteria including minimum count thresholds and proportions.

    Parameters:
        y (np.array): Expression data for the genes.
        design (np.array, optional): Design matrix for the samples if available.
        group (np.array, optional): Group information for samples.
        lib_size (np.array, optional): Library sizes for the samples.
        min_count (int): Minimum count threshold for including a gene.
        min_total_count (int): Minimum total count across all samples for a gene.
        large_n (int): Cutoff for considering a sample 'large'.
        min_prop (float): Minimum proportion used in calculations for large sample consideration.

    Returns:
        np.array: Boolean array indicating which genes to keep.
    """    
    y = np.asarray(y)
    
    if y.dtype != 'float32':
        raise ValueError("y is not a numeric matrix")

    if lib_size is None:
        lib_size = np.sum(y, axis=0)

    if group is None:
        if design is None:
            print("No group or design set. Assuming all samples belong to one group.")
            min_sample_size = y.shape[1]
        else:
            hat_values = np.linalg.norm(np.dot(design, np.linalg.pinv(design)), axis=1)
            min_sample_size = 1 / np.max(hat_values)
    else:
        _, n = np.unique(group, return_counts=True)
        min_sample_size = np.min(n[n > 0])

    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop

    median_lib_size = np.median(lib_size)
    cpm_cutoff = min_count / median_lib_size * 1e6

    cpm_values = custom_cpm(y, lib_size)
    keep_cpm = np.sum(cpm_values >= cpm_cutoff, axis=1) >= (min_sample_size - np.finfo(float).eps)
    keep_total_count = np.sum(y, axis=1) >= (min_total_count - np.finfo(float).eps)

    return np.logical_and(keep_cpm, keep_total_count)