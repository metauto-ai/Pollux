import os
import logging
import datasets

logger = logging.getLogger()

def HFDataLoad(data_name: str, cache_dir: str) -> str:

    if os.path.exists(
        os.path.join(cache_dir, data_name.split("/")[-1])
    ):
        data = datasets.load_dataset(
            path=data_name, cache_dir=os.path.join(cache_dir, data_name.split("/")[-1])
        )
        logger.info(f'Dataset "{data_name}" loaded from local cache.')
    else:
        data = datasets.load_dataset(
            path=data_name, cache_dir=os.path.join(cache_dir, data_name.split("/")[-1])
        )
        logger.info(f'Dataset "{data_name}" downloaded from huggingface and cached.')
        import stat

        os.chmod(
            os.path.join(cache_dir, data_name.split("/")[-1]),
            stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
        )

    return data
