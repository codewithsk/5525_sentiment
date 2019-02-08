#PBS -N node2vec
#PBS -A PAS1425
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=40
#PBS -j oe

cd /users/PAS1197/osu9208/CSE5525/5525_sentiment

tar -xvzf data.tgz
