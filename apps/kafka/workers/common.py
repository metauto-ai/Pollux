import multiprocessing
import threading
import time
import yaml
import wandb


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_wandb(config: dict, run_name: str):
    """Initialize wandb with common settings"""
    name = f"{run_name}-{int(time.time())}"
    wandb.init(
        project=config.get("wandb", {}).get("project", "kafka-pipeline"),
        entity=config.get("wandb", {}).get("entity", None),
        config=config,
        name=name,
    )
    return wandb.run


def print_counter():
    total_counter = multiprocessing.Value('i', 0)
    counter = multiprocessing.Value('i', 0)

    def counter_daemon():
        print_every = 10
        step = 0
        while True:
            time.sleep(print_every)
            with counter.get_lock():
                counter_value = counter.value
                counter.value = 0
            with total_counter.get_lock():
                total_counter.value += counter_value
            
            # Log to wandb if initialized
            if wandb.run is not None and total_counter.value > 0:
                wandb.log({
                    "total_documents": total_counter.value,
                    "documents_per_second": counter_value / print_every,
                    "step": step
                })
                step += print_every
            
            print (f"---------------------------------------------------------------------------------------------")
            print (f"{total_counter.value} documents processed: {counter_value / print_every:.2f} documents per second")
            print (f"---------------------------------------------------------------------------------------------")

    counter_thread = threading.Thread(target=counter_daemon, daemon=True)
    counter_thread.start()

    return counter
