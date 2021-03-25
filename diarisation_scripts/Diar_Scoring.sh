#cd /mnt/lustre/users/lvanwyk1/Diarization_Utils/diariation_scripts

for window_length in $(ls $1)
do
	echo $window_length
	for step_length in $(ls $1/$window_length)
	do
		echo $step_length
		qsub -v fileids=$fileids,ref_rttm=$ref_rttm,hyp_rttm=$1/$window_length/$step_length pbs.Diar_Scoring
	done
	sleep 30
done

echo $ref_rttm
