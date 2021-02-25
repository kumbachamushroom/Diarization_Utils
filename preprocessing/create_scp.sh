#Script takes directory of wav files and create wav.scp file of format <file-id> <filepath>
# Arguments:
# 	<wav-dir> Directory to wav files
#	

for file in $(find $1 -type f -name "*.wav")
do
	file_id=$(echo $(basename $file) | rev | cut -d"." -f2- | rev)
	filepath=$file
	echo $file_id	$filepath
done >> $1/wav.scp
