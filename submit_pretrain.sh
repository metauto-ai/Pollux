#!/bin/bash
# Helper script to submit and monitor SLURM jobs

# Make sure scripts are executable
chmod +x setup_shared_env.sh
chmod +x train_castor.slurm
chmod +x diagnose_flash_attn.py

# Check if diagnostic mode is requested
if [ "$1" == "diagnose" ]; then
    echo "Running flash-attention diagnostic tool..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate /mnt/pollux/environments/pollux_env
    python diagnose_flash_attn.py
    exit 0
fi

# Check if a custom partition is provided
if [ -n "$1" ] && [ "$1" != "diagnose" ]; then
    PARTITION="$1"
    echo "Using custom partition: $PARTITION"
    # Submit the job with the specified partition
    JOB_ID=$(sbatch --parsable --partition="$PARTITION" train_castor.slurm)
else
    # Submit the job with the default partition in the script
    JOB_ID=$(sbatch --parsable train_castor.slurm)
fi

echo "Job submitted with ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
echo "View logs with: tail -f ${JOB_ID}.out"
echo ""
echo "Quick commands:"
echo "---------------"
echo "View job status: scontrol show job $JOB_ID"
echo "Cancel job: scancel $JOB_ID"
echo "View resource usage: sstat --format=AveCPU,AveRSS,AveVMSize,MaxRSS,MaxVMSize -j $JOB_ID"
echo "Run diagnostic tool: ./submit_job.sh diagnose"

# Watch the job status
echo ""
echo "Watching job status (press Ctrl+C to exit):"
watch -n 10 squeue -j "$JOB_ID" 