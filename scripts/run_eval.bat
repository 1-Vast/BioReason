@echo off
echo BioReason Evaluation
python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
