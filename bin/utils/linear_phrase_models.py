# Classics
import os
import glob
import numpy as np
import pandas as pd

# Logit model library
import statsmodels.api as sm

# Local functions
import sys
sys.path.append('/cellar/users/aklie/projects/EUGENE/bin/')
from project_utils import split_train_test, standardize_features, classification_report

# Define dataset
DATASET = "/cellar/users/aklie/projects/EUGENE/data/2021_OLS_Library"  # Which dataset to look at
FEATURES = "phrase_encodings"  # What features to use to train the model

# Define fitting procedure
SPLIT = 0.9
SUBSET = None
HYPERPARAM = "baseline"
MODEL = "LR"

files = np.array(glob.glob("{}/{}/*False.tsv".format(DATASET, FEATURES)))
files = files[np.argsort([len(file) for file in files])]
for i, file in enumerate(files):
    if os.path.getsize(file) <= 1e9:
        filename = file.split("/")[-1].replace(".tsv", "")
        print(filename + " STARTED")
        outdir = "../results/phrase/{}/{}".format(MODEL, filename)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Load the dataset
        data = pd.read_csv("{}/{}/{}.tsv".format(DATASET, FEATURES, filename), sep="\t", low_memory=False)
        x_cols = data.columns[data.columns.str.contains("((e.*|E.*|g.*|G.*)_)+")]

        # Use Joe's labels
        #data["LABEL"] = np.nan
        #data["LABEL"] = np.where(data["NONFUNC-<0.625"]=="True", 0, np.nan)
        #data["LABEL"] = np.where(data["FUNC->0.685"]=="True", 1, data["LABEL"])
        
        # Use my own
        data["LABEL"] = np.nan
        data['ACT_SumRNA_LenDNA_LOG2_NORM'] = data['ACT_SumRNA_LenDNA_LOG2_NORM'].replace({"ND":np.nan})
        data["ACT_SumRNA_LenDNA_LOG2_NORM"] = data["ACT_SumRNA_LenDNA_LOG2_NORM"].astype(float)
        data.loc[data['ACT_SumRNA_LenDNA_LOG2_NORM'] <= 0.600, "LABEL"] = 0
        data.loc[data['ACT_SumRNA_LenDNA_LOG2_NORM'] >= 0.685, "LABEL"] = 1
        if 'ACT_SumRNA_LenDNA_LOG2_NORM' in x_cols:
            x_cols = x_cols.drop('ACT_SumRNA_LenDNA_LOG2_NORM')

        # Grab dependent variable and regressors
        training_data = data[~data["LABEL"].isna()]
        X = training_data[x_cols].values
        y = training_data["LABEL"].values

        # Feature preprocessing
        X_train, X_test, y_train, y_test = split_train_test(X, y, split=SPLIT, subset=SUBSET, rand_state=13, shuf=True)
        #X_train, X_test = standardize_features(X_train, X_test)

        # Save as pandas format
        X_train = pd.DataFrame(X_train, columns=x_cols)
        y_train = pd.Series(y_train, name="LABEL")

        # Fit a model
        try:
            log_reg = sm.Logit(y_train, sm.tools.add_constant(X_train, has_constant="add"))
            #res = log_reg.fit_regularized(alpha=0.1)
            res = log_reg.fit()

        except:
            print("Encountered error, skipping this fit")
            continue


        # Get the classification results
        prob_thresh = 0.5
        y_tr_probs = res.predict(sm.tools.add_constant(X_train, has_constant="add"))
        y_probs = res.predict(sm.tools.add_constant(X_test, has_constant="add"))
        y_tr_preds = (y_tr_probs >= prob_thresh).astype(int)
        y_preds = (y_probs >= prob_thresh).astype(int)

        # Generate a report
        classification_report(out_path=outdir,
            train_X=X_train, test_X=X_test, 
            train_y=y_train, test_y=y_test,
            train_preds=y_tr_preds, test_preds=y_preds,
            train_probs=y_tr_probs, test_probs=y_probs)

        # Save inference statistics
        pd.DataFrame(data={"coefficient": res.params, "p_value": res.pvalues}).to_csv(os.path.join(outdir, "coefficients.tsv"), sep="\t")
        print("COMPLETED")