
import os
import logging
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Iterable, Optional, Tuple

logger = logging.getLogger("eugene")


def single_sample_recipe(
    frag_file: str,
    outdir_path: str,
    sample_name: str = None,
    min_load_num_fragments: int = 500,
    sorted_by_barcode: bool = True,
    chunk_size: int = 2000,
    save_intermediate: bool = False,
    min_tsse: int = 4,
    min_num_fragments: int = 1000,
    max_num_fragments: int = None,
    additional_doublets: str = None,
    metadata: pd.DataFrame = None,
    bin_size: int = 500,
    num_features: int = 50000,
    blacklist_path=None,
    clustering_resolution: float = 1.0,
    gene_activity: bool = True,
):

    # Log snapATAC version
    logger.info(f"Running standard single sample workflow with snapATAC version {snap.__version__}")

    # Load in from fragment file into memory
    logger.info("Loading in data using `import_data` function without file backing")
    adata = snap.pp.import_data(
        fragment_file=frag_file,
        chrom_sizes=snap.genome.hg38,
        min_num_fragments=min_load_num_fragments,
        sorted_by_barcode=sorted_by_barcode,
        chunk_size=chunk_size,
        n_jobs=-1,
    )

    # Add sample name to barcode with # in between
    if sample_name is not None:
        logger.info(f"Adding sample name {sample_name} to barcode")
        logger.info(f"Before: {adata.obs.index[0]}")
        adata.obs.index = sample_name + "#" + adata.obs.index
        logger.info(f"After: {adata.obs.index[0]}")
        logger.info("If passing in metadata, make sure to add sample name to barcode column as well")

    # Plot fragment size distribution
    logger.info("Plotting fragment size distribution")
    snap.pl.frag_size_distr(adata, interactive=False, out_file=os.path.join(outdir_path, "frag_size_distr.png"))

    # Plot TSSe distribution vs number of fragments
    logger.info("Plotting TSSe distribution vs number of fragments")
    snap.metrics.tsse(adata, snap.genome.hg38)
    snap.pl.tsse(adata, interactive=False, out_file=os.path.join(outdir_path, "nfrag_vs_tsse.png"))

    # Save the processed data
    if save_intermediate:
        logger.info("Saving the qc data prior to filtering")
        adata.write(os.path.join(outdir_path, f"qc.h5ad"))

    # Filter out low quality cells
    logger.info(f"Filtering out low quality cells with tsse<{min_tsse} and min_num_fragments<{min_num_fragments}, max_num_fragments>{max_num_fragments}")
    adata.obs["log_n_fragment"] = np.log10(adata.obs["n_fragment"] + 1)
    snap.pp.filter_cells(adata, min_tsse=min_tsse, min_counts=min_num_fragments, max_counts=max_num_fragments)

    # Report number of cells after filtering
    logger.info(f"Number of cells after filtering: {adata.shape[0]}")

    # Add a 5kb tile matrix
    logger.info(f"Adding a {bin_size}bp tile matrix")
    snap.pp.add_tile_matrix(adata, bin_size=bin_size)

    # Select the top accessible features
    logger.info(f"Selecting the top {num_features} accessible features")
    snap.pp.select_features(adata, n_features=num_features, blacklist=blacklist_path)

    # Run scrublet
    logger.info("Running scrublet")
    snap.pp.scrublet(adata)

    # Filter out doublets
    logger.info("Filtering out doublets")
    snap.pp.filter_doublets(adata)

    # Filter out additional doublets if passed in
    if additional_doublets is not None:
        logger.info(f"Filtering out additional doublets from {additional_doublets}")
        additional_doublets = pd.read_csv(additional_doublets, header=None, index_col=0).index
        adata = adata[~adata.obs.index.isin(additional_doublets)]
    
    # Report number of cells after filtering
    logger.info(f"Number of cells after filtering doublets: {adata.shape[0]}")
    
    # Add in metadata if passed in
    logger.info("Subsetting data to metadata if passed in")
    if metadata is not None:
        num_intersecting_cells = len(set(metadata.index).intersection(set(adata.obs.index)))
        logger.info(f"Number of cells found in metadata: {num_intersecting_cells}")

        # Subset the object to only include cells in the metadata
        adata = adata[adata.obs.index.isin(metadata.index)]
        
        # Add in the metadata
        adata.obs = adata.obs.merge(metadata, left_index=True, right_index=True, suffixes=("", "_rna"))

        # Check
        logger.info(f"Number of cells after subsetting to metadata: {adata.shape[0]}")

    # Run the spectral embedding
    logger.info("Running spectral embedding")
    snap.tl.spectral(adata)

    # Plot first spectral embedding against log_n_fragment    
    with plt.rc_context({"figure.figsize": (5, 5)}):
        sc.pl.embedding(basis="X_spectral", adata=adata, color="log_n_fragment", show=False)
        plt.savefig(os.path.join(outdir_path, "spectral_embedding.png"))
        plt.close()

    # Run UMAP
    logger.info("Running UMAP")
    snap.tl.umap(adata, use_dims=list(range(1, adata.obsm["X_spectral"].shape[1])))

    # Find nearest neighbor graph
    logger.info("Finding nearest neighbor graph")
    snap.pp.knn(adata, use_rep="X_spectral", use_dims=list(range(1, adata.obsm["X_spectral"].shape[1])))

    # Cluster data
    logger.info(f"Clustering data using Leiden algorithm with resolution {clustering_resolution}")
    snap.tl.leiden(adata, resolution=clustering_resolution, key_added=f"leiden_{clustering_resolution}")

    # Plot the UMAP with clusters
    logger.info("Plotting UMAP with clusters")
    with plt.rc_context({"figure.figsize": (5, 5)}):
        sc.pl.umap(adata, color=["log_n_fragment", "tsse", f"leiden_{clustering_resolution}"], show=False)
        plt.savefig(os.path.join(outdir_path, "umap.png"))
        plt.close()

    # Save updated data
    logger.info("Saving clustered data")
    adata.write(os.path.join(outdir_path, f"clustered.h5ad"))
    adata.obs.to_csv(os.path.join(outdir_path, f"cell_metadata.tsv"), sep="\t")

    # Create a gene matrix
    if gene_activity:
        # Creating gene matrix
        logger.info("Creating gene activity matrix")
        gene_matrix = snap.pp.make_gene_matrix(adata=adata, gene_anno=snap.genome.hg38)

        # Clean up the gene matrix
        logger.info("Filtering and normalizing the gene activity matrix")
        sc.pp.filter_genes(gene_matrix, min_cells=3)
        sc.pp.normalize_total(gene_matrix)
        sc.pp.log1p(gene_matrix)

        # Run MAGIC
        logger.info("Running MAGIC on the gene activity matrix for imputation")
        sc.external.pp.magic(gene_matrix, solver="approximate")

        # Transfer the UMAP from the original data to the gene matrix
        gene_matrix.obsm["X_umap"] = adata.obsm["X_umap"]

        # Save the gene matrix
        logger.info("Saving gene activity matrix")
        gene_matrix.write(os.path.join(outdir_path, f"gene_matrix.h5ad"))


def integrate_recipe(
    input_h5ad_paths: list,
    sample_ids: list,
    outdir_path: str,
    output_prefix: Optional[str] = "integrated",
    annotation_key: Optional[str] = None,
    barcodes_path: Optional[str] = None,
    n_features: int = 50000,
    clustering_resolution: float = 1.0,
    make_gene_matrix: bool = False,
    filter_genes: int = 3,
):
    # Log snapATAC version
    logger.info(f"Running standard integration workflow with snapATAC version {snap.__version__}")

    # If sample ids are not provided, use the file names
    if sample_ids is None:
        logger.info("Sample ids not provided. Using file names.")
        sample_ids = [os.path.basename(file).split(".")[0] for file in input_h5ad_paths]
    
    # Read in each h5ad with scanpy delete the X_spectral and X_umap from the obsm of the AnnData and resave with new name
    logger.info("Deleting X_spectral and X_umap from obsm of AnnData and resaving.")
    cell_bcs = []
    cell_ids = []
    for path in input_h5ad_paths:
        adata = sc.read_h5ad(path)
        del adata.obsm["X_spectral"]
        del adata.obsm["X_umap"]
        adata.write_h5ad(path.replace(".h5ad", "_obsm_delete.h5ad"))
        if annotation_key is not None:
            cell_ids.extend(adata.obs[annotation_key].tolist())
            cell_bcs.extend(adata.obs_names.tolist())
    if annotation_key:
        cell_id_map = pd.Series(cell_ids, index=cell_bcs)

    # Update the input h5ad paths to the new ones
    input_h5ad_paths = [file.replace(".h5ad", "_obsm_delete.h5ad") for file in input_h5ad_paths]
    logger.info(f"Updated input h5ad paths: {input_h5ad_paths}")
    
    # Read in barcodes file if provided
    if barcodes_path is not None:
        logger.info(f"Reading in barcodes file from {barcodes_path}")
        if barcodes_path.endswith(".csv"):
            barcodes = pd.read_csv(barcodes_path, header=None, index_col=0)
        elif barcodes_path.endswith(".tsv") | barcodes_path.endswith(".txt"):
            barcodes = pd.read_csv(barcodes_path, header=None, index_col=0, sep="\t")
        else:
            raise ValueError("Barcodes file must be a .csv or .tsv file.")
        barcodes = barcodes.index.tolist()
        logger.info(f"Barcodes file read in with {len(barcodes)} barcodes.")
        logger.info(f"First few barcodes: {barcodes[:5]}")

    # Create the AnnDataset
    adata_atac_merged_list = []
    for _, h5ad_file in enumerate(input_h5ad_paths):
        adata_atac = snap.read(h5ad_file)
        if barcodes_path is not None:
            logger.info(f"Subsetting AnnDataset to barcodes in {barcodes_path}")
            adata_atac.subset(obs_indices=np.where(pd.Index(adata_atac.obs_names).isin(barcodes))[0])
            logger.info(f"Number of cells after filtering: {adata_atac.shape[0]}")
        adata_atac_merged_list.append(adata_atac)
    adatas = [(name, adata) for name, adata in zip(sample_ids, adata_atac_merged_list)]

    # Merge into one object
    logger.info(f"Creating AnnDataset from {adatas} samples.")
    adata_atac_merged = snap.AnnDataSet(
        adatas=adatas,
        filename=os.path.join(outdir_path, f"{output_prefix}.h5ads")
    )
    logger.info(f"AnnDataset created at {os.path.join(outdir_path, f'{output_prefix}.h5ads')}")

    # Close all the backed anndatas
    logger.info("Closing all the backed anndatas.")
    for adata_atac in adata_atac_merged_list:
        adata_atac.close()
    adata_atac_merged.close()

    # Read in the merged AnnDataset
    adata_atac_merged = snap.read_dataset(os.path.join(outdir_path, f"{output_prefix}.h5ads"))
    if annotation_key is not None:
        logger.info(f"First few merged adata cell ids: {adata_atac_merged.obs_names[:5]}")
        logger.info(f"First few cell id map entries: {cell_id_map.index[:5]}")
        mapped_cell_ids = cell_id_map[adata_atac_merged.obs_names].values.tolist()
        adata_atac_merged.obs[annotation_key] = mapped_cell_ids
    
    # Select features on the merged dataset
    logger.info(f"Selecting {n_features} features.")
    snap.pp.select_features(adata_atac_merged, n_features=n_features)

    # Spectral embedding
    logger.info("Performing spectral embedding")
    snap.tl.spectral(adata_atac_merged)

    # Run UMAP
    logger.info("Running UMAP")
    snap.tl.umap(adata_atac_merged, use_dims=list(range(1, adata_atac_merged.obsm["X_spectral"].shape[1])))

    # Clustering
    logger.info("Performing clustering")
    snap.pp.knn(adata_atac_merged, use_rep="X_spectral", use_dims=list(range(1, adata_atac_merged.obsm["X_spectral"].shape[1])))
    snap.tl.leiden(adata_atac_merged, resolution=clustering_resolution, key_added=f"leiden_{clustering_resolution}")

    if make_gene_matrix:

        # Create gene matrix
        logger.info("Creating gene matrix")
        gene_matrix = snap.pp.make_gene_matrix(
            adata=adata_atac_merged,
            gene_anno=snap.genome.hg38
        )

        # Clean up the gene matrix
        sc.pp.filter_genes(gene_matrix, min_cells=filter_genes)
        sc.pp.normalize_total(gene_matrix)
        sc.pp.log1p(gene_matrix)
        
        # Perform imputation with MAGIC
        logger.info("Performing MAGIC imputation")
        sc.external.pp.magic(gene_matrix, solver="approximate")

        # Copy over UMAP embedding
        gene_matrix.obsm["X_umap"] = adata_atac_merged.obsm["X_umap"]

        # Write the gene matrix
        logger.info("Writing gene matrix")
        gene_matrix.write(os.path.join(outdir_path, f"{output_prefix}_gene_matrix.h5ad"))

    # Turn into in memory AnnData
    logger.info("Turning into in memory AnnData")
    adata_mem = adata_atac_merged.to_adata()

    # Plot first spectral embedding against log_n_fragment
    with plt.rc_context({"figure.figsize": (5, 5)}):
        sc.pl.embedding(adata=adata_mem, basis="X_spectral", color="log_n_fragment", show=False)
        plt.savefig(os.path.join(outdir_path, "spectral_embedding.png"))
        plt.close()
        
    # Plot the UMAP with clusters
    logger.info("Plotting UMAP with clusters")
    with plt.rc_context({"figure.figsize": (5, 5)}):
        sc.pl.umap(adata_mem, color=["log_n_fragment", "tsse", "sample" f"leiden_{clustering_resolution}"], show=False, ncols=2)
        plt.savefig(os.path.join(outdir_path, "umap.png"))
        plt.close()

    # Close the file
    adata_atac_merged.close()

    # Print completion message
    logger.info("Successfully completed integration of SnapATAC2 samples.")
