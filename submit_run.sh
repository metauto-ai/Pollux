# Full customization
./submit_training.sh \
    -n 2 \
    -p dev \
    -e pollux \
    -c apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml \
    -j castor-multinode \
    -t 168:00:00