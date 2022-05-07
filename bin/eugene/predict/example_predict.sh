python hybrid.py predict \
    --seed_everything 13 \
    --config ../../config/models/simple/classification/ss_bin-clf_hybrid.yaml \
    --ckpt_path ../../results/simple/classification/sshybrid/2022_04_23_NPY_Baseline/checkpoints/epoch\=25-step\=4523.ckpt \
    --config ../../config/data/test/All_Genomic_Sequences_TSV_bin_clf.yaml \
    --trainer.callbacks=PredictionWriter \
    --trainer.callbacks.output_dir ../../results/simple/classification/sshybrid/2022_04_23_NPY_Baseline/predictions/All_Genomic_Sequences_ \
    --trainer.logger False \
    --trainer.gpus 1