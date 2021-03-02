export src_data_dir=$1
export sad_nnet_dir=~/Diarization_Utils/preprocessing/Aspire_TDNN_SAD/exp/segmentation_1a/tdnn_stats_asr_sad_1a
export exp_dir=$2
mfcc_dir=$exp_dir/mfcc_hires
mkdir -p $mfcc_dir
export work_dir=$exp_dir/work
mkdir -p $work_dir
export out_dir=$exp_dir/out
mkdir -p $out_dir

#cd $KALDI_ROOT/egs/aspire/s5

#bash  steps/segmentation/detect_speech_activity.sh  --extra-left-context 79 --extra-right-context 21 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150 --mfcc-config conf/mfcc_hires.conf $src_data_dur $sad_nnet_dir $mfcc_dir $work_dir $out_dir



