export nnUNet_N_proc_DA=36
export nnUNet_codebase="/home/changbo/Segmentation/nnUNet_V1"
export nnUNet_raw_data_base="/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/changbo/Segmentation/3D-TransUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/changbo/Segmentation/3D-TransUNet/results/"
export CUDA_VISIBLE_DEVICES=0

config='/home/changbo/Segmentation/3D-TransUNet/configs/ACDC/decoder_only.yaml'
subset='/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/imagesTs'
save_folder='/home/changbo/Segmentation/3D-TransUNet/results/inference/Task027_ACDC/decoderonly'
NUM_GPUS=0


fold='all'
extra=${@:3}
echo "inference: fold ${fold} on gpu ${gpu}, extra ${extra}"
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0 \
python3 /home/changbo/Segmentation/3D-TransUNet/inference.py --config=${config} \
		--fold=${fold} --raw_data_folder ${subset} \
		--save_folder=${save_folder}/fold_${fold} ${extra}


