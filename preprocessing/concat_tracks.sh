#Concat all wav tracks in dir ($1) and saves it as ($2)
#Usage: concat_tracks.sh <audio-dir> <save-dir>
echo $1
echo $2

sox $(ls $1/*.wav | head -50) $2
