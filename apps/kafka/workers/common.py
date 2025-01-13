import multiprocessing
import threading
import time
import yaml


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_counter():
    total_counter = multiprocessing.Value('i', 0)
    counter = multiprocessing.Value('i', 0)

    def counter_daemon():
        print_every = 60
        while True:
            time.sleep(print_every)
            with counter.get_lock():
                counter_value = counter.value
                counter.value = 0
            with total_counter.get_lock():
                total_counter.value += counter_value
            print (f"---------------------------------------------------------------------------------------------")
            print (f"{total_counter.value} documents processed: {counter_value / print_every:.2f} documents per second")
            print (f"---------------------------------------------------------------------------------------------")

    counter_thread = threading.Thread(target=counter_daemon, daemon=True)
    counter_thread.start()

    return counter
