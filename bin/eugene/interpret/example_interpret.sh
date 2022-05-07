python interpret.py score \
    --model ../results/simple/classification/sshybrid/2022_04_23_NPY_Baseline/checkpoints/epoch\=25-step\=4523.ckpt \
    --model_type hybrid \
    --data ../config/data/test/All_Genomic_Sequences_TSV_bin_clf.yaml \
    --output ../results/simple/classification/sshybrid/2022_04_23_NPY_Baseline/interpretations/All_Genomic_Sequences