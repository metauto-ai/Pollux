"""
python -m apps.main.test
"""
from apps.Simple_DiT.data import  create_dummy_dataloader, create_imagenet_dataloader,DataArgs, may_download_image_dataset
from lingua.transformer import precompute_freqs_cis
from apps.Simple_DiT.transformer import precompute_2d_freqs_cls, DiffusionTransformer, DiffusionTransformerArgs
from apps.main.model import DiffuserVAEArgs, DiffuserVAE, LatentTransformer, ModelArgs
from apps.Simple_DiT.schedulers import SchedulerArgs, RectFlow
import logging
from torchvision.utils import save_image
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
    dit_args = DiffusionTransformerArgs(
        dim= 2048,
        n_layers= 25,
        n_heads= 16,
        patch_size=2,
        in_channels=16,
    )
    dataloader = create_dummy_dataloader(batch_size=16,num_classes=dit_args.num_classes,image_size=(16,32,32))
    DiT = DiffusionTransformer(dit_args).cuda()
    vae_args = DiffuserVAEArgs()
    schedulers_arg = SchedulerArgs()
    scheduler = RectFlow(schedulers_arg)
    for class_idx, time_step, image in dataloader:
        class_idx = class_idx.cuda()
        time_step = time_step.cuda()
        image = image.cuda()
        noised_x, t, target = scheduler.sample_noised_input(image)
        output = DiT(x=noised_x,time_steps=t,context=class_idx)
    
    model_args = ModelArgs()
    model_args.transformer = dit_args
    model_args.vae = vae_args
    model_args.scheduler = schedulers_arg
    model = LatentTransformer(model_args).cuda()
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
            data[k]=data[k].cuda()
        model(data)
        break
