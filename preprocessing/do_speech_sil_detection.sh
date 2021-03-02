cd $KALDI_ROOT/egs/aspire/s5

test_stage=-10
test_nj=30

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi
  # Use left and right context options that were used when training
  # the chain nnet
  # Increase sil-scale to predict more silence
local/nnet3/segment_and_decode.sh --stage $test_stage \
    --decode-num-jobs $test_nj --affix "${test_affix}" \
    --sad-opts "$sad_opts" \
    --sad-graph-opts "--min-silence-duration=0.03 --min-speech-duration=0.3 --max-speech-duration=10.0" --sad-priors-opts "--sil-scale=0.1" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context 50 \
    --extra-right-context 0 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
   --sub-speaker-frames 6000 --max-count 75 \
   --decode-opts "--min-active 1000" \
dev_aspire $sad_nnet_dir $sad_nnet_dir data/lang $chain_dir/graph_pp $chain_dir
