import inspect
from pprint import pprint

from . import humaneval, mbpp

TASK_REGISTRY = {
    **humaneval.create_all_tasks(),  
    "mbpp": mbpp.MBPP,               
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if args is None:
            args = type("Args", (), {})()  
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters and hasattr(args, "prompt"):
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters and hasattr(args, "load_data_path"):
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
