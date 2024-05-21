export nnUNet_N_proc_DA=36
export nnUNet_codebase="/home/changbo/Segmentation/nnUNet_V1"
export nnUNet_raw_data_base="/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/changbo/Segmentation/3D-TransUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/changbo/Segmentation/3D-TransUNet/results/"
export CUDA_VISIBLE_DEVICES=0

config='/home/changbo/Segmentation/3D-TransUNet/configs/ACDC/decoder_only.yaml'
subset='/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/imagesTr'
save_folder='/home/changbo/Segmentation/3D-TransUNet/results/inference/Task027_ACDC/decoderonly'
raw_data_dir='/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/imagesTr/'
NUM_GPUS=0


fold='all'  # Assign the variable fold to the first argument passed during script execution
pred_dir=${2:-${save_folder}/fold_${fold}/}
extra=${@:3}

echo "compute_metric: fold ${fold}, extra ${extra}"
python3 /home/changbo/Segmentation/3D-TransUNet/measure_dice.py \
		--config=${config} --fold=${fold} \
		--raw_data_dir=${raw_data_dir} \
		--pred_dir=${pred_dir} ${extra}
