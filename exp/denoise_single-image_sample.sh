
python test_denoise.py --config ./src/config/C2N_DnCNN.yml --ckpt ./ckpt/DnCNN-SIDD_to_SIDD-on_SIDD.ckpt --mode single --data ./data/test/SIDD1.png --gpu 0

# python test_denoise.py --config ./src/config/C2N_DIDN.yml --ckpt ./ckpt/DIDN-SIDD_to_SIDD-on_SIDD.ckpt --mode single --data ./data/test/SIDD1.png --gpu 0

# python test_denoise.py --config ./src/config/C2N_DIDN.yml --ckpt ./ckpt/DIDN-DND_to_SIDD-on_SIDD.ckpt --mode single --data ./data/test/DND_sample.png --gpu 0