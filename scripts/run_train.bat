@echo off
echo BioReason Training - Stage 1
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1 --device cuda
