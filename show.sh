seed="11,12,13,14,15,16"
trunc="0.5"
seeds_s=0
seeds_e=5
#python generate.py --outdir=tmp_show --trunc="${trunc}" --seeds="${seed}" --network=../../exp_car_ffful0/00000-comp-paper512-stylenerf_ffhq-noaug/network-snapshot-004810.pkl --render-program="rotation_camera"
python create_dataset.py --outdir=output/dataset_trunc_new --trunc=0.5 --seeds_e=5 --n_steps=16 --network=./car_model.pkl --render-program="rotation_camera"
python create_dataset.py --outdir=output/dataset_test --trunc="${trunc}" --seeds_s="${seeds_s}" --seeds_e="${seeds_e}" --network=./car_model.pkl --render-program="rotation_camera"