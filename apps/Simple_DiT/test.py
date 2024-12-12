from apps.Simple_DiT.transformer import DiTransformer, DiTransformerArgs
from apps.Simple_DiT.data import  create_dummy_dataloader, create_imagenet_dataloader,DataArgs, may_download_image_dataset
from lingua.transformer import precompute_freqs_cis
from apps.Simple_DiT.transformer import precompute_2d_freqs_cls
from apps.Simple_DiT.schedulers import SchedulerArgs, RectFlow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
    handlers=[
        logging.StreamHandler()         # Also log to console
    ]
)



if __name__ == '__main__':
    # may_download_image_dataset('/mnt/data/imagenet')
    freqs_1d_cls = precompute_freqs_cis(512, 1024)
    freqs_2d_cls = precompute_2d_freqs_cls(512, 1024)
    args = DiTransformerArgs(
        dim= 2048,
        ffn_dim_multiplier= 1.5,
        multiple_of= 256,
        n_heads= 32,
        n_kv_heads= 8,
        n_layers= 16,
        ada_dim= 512,
        patch_size= 16,
        in_channels= 3,
        out_channels= 3,
        tmb_size= 256,
        cfg_drop_ratio= 0.1,
        num_classes= 1000,
        max_seqlen= 1000,
    )
    dataloader = create_dummy_dataloader(batch_size=16,num_classes=args.num_classes)
    model = DiTransformer(args)
    model.init_weights('/mnt/data/Llama-3.2-1B/original/consolidated.00.pth')
    model.cuda()
    schedulers_arg = SchedulerArgs()
    scheduler = RectFlow(schedulers_arg)
    for class_idx, time_step, image in dataloader:
        class_idx = class_idx.cuda()
        time_step = time_step.cuda()
        image = image.cuda()
        noised_x, t, target = scheduler.sample_noised_input(image)
        output = model(x=noised_x,time_steps=t,context=class_idx)
    data_arg = DataArgs(
        root_dir='/mnt/data/imagenet',
        image_size=256,
        num_workers=8,
        batch_size=16,
    )
    train_loader = create_imagenet_dataloader(
                                            shard_id=0,
                                            num_shards=4, 
                                            args = data_arg
                                            )
    for data in train_loader:
        assert isinstance(data, dict)
        for k in data.keys():
            logging.info(f"[{k}]'s shape : {data[k].size()}")
        break