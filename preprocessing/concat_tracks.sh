#Concat all wav tracks in dir ($1) and saves it as ($2)

sox $(ls $1/*.wav | head -50) $2
