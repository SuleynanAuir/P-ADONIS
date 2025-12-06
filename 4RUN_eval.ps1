conda activate pean

# Run evaluation: Random sample 20 images, 5 images per group (4 groups total)
# Parameters:
# --random_sample 20 = randomly sample 20 images from test set
# --samples_per_grid 5 = 5 images per visualization group
# --num_grids 4 = generate 4 combined grid visualizations (20/5=4)
python run_eval_pean.py --pean_ckpt ./ckpt/PEAN_final.pth --tpem_ckpt ./ckpt/TPEM_final.pth --subset easy --random_sample 20 --samples_per_grid 5 --num_grids 4 --out_dir ./eval_results --srb 1
# python run_eval_pean.py --pean_ckpt ./ckpt/PEAN_final.pth --tpem_ckpt ./ckpt/TPEM_final.pth --out_dir ./eval_results --batch_size 1 --srb 1