
export nnUNet_N_proc_DA=36
export nnUNet_codebase="/home/changbo/Segmentation/nnUNet_V1"
export nnUNet_raw_data_base="/home/changbo/Segmentation/3D-TransUNet/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/changbo/Segmentation/3D-TransUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/changbo/Segmentation/3D-TransUNet/results/"
export CUDA_VISIBLE_DEVICES=0

CONFIG=$1
echo $CONFIG

fold=0
echo "run on fold: ${fold}"
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0 \
    python3 /home/changbo/Segmentation/3D-TransUNet/train.py --fold='0'\
    --config='/home/changbo/Segmentation/3D-TransUNet/configs/ACDC/decoder_only.yaml' --resume='local_latest'