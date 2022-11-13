from pathlib import Path
import importlib


def check_module():
    optional_package = ["pytorch"]
    exclude_package = ["SciencePlots"]
    module = []
    for i in open("requirements.txt").readlines():
        name = i.strip().split("=")[0]
        if name not in exclude_package:
            module.append(name)
            try:
                mod = importlib.import_module(name)
                print(f"Import {name}{mod.__version__} successfully.")
            except ModuleNotFoundError:
                if name in optional_package:
                    print(f"{name} module is not found! Optionally install it.")
                else:
                    print(f"{name} module is not found! One should install it.")
    return module


if __name__ == "__main__":
    module = check_module()
    print("Packages include:", module)
