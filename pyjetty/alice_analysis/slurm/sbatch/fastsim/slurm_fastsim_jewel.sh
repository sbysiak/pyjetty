#! /bin/bash

#SBATCH --job-name=fastsim-jewel
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=test
#SBATCH --time=1:00:00
#SBATCH --array=1-800
#SBATCH --output=/rstorage/generators/jewel_alice/tree_fastsim/slurm-%A_%a.out

FILE_PATHS='/rstorage/generators/jewel_alice/tree_gen/746611/files.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

# Currently we have 8 nodes * 20 cores active
FILES_PER_JOB=1
echo "Files per job: $FILES_PER_JOB"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START=$START"
echo "STOP=$STOP"

for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  srun process_fastsim_jewel.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
done