@echo off
echo BioReason Inference
python main.py infer --config config/default.yaml --checkpoint output/stage3/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
