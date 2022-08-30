from pathlib import Path
import importlib
from pyAnalysis.screen_output.custom_print import print_color, Color


def check_module():
    optional_package = ['freud']
    def get_module():
        module = []
        path = Path('./')
        for i in path.rglob('*.py'):
            with open(i, 'r', encoding='UTF-8') as op:
                file = op.readlines()
            if file:
                for i in file:
                    if len(i.split()) > 0:
                        if i.split()[0] in ['import', 'from']:
                        #if i[:6] == 'import' or i[:4] == 'from':
                            module.append(i.split()[1].split('.')[0])
        module = list(set(module))
        #module.remove('pyAnalysis')
        return module
    module = get_module()
    for name in module:
        try:
            importlib.import_module(name)
            print_color(f'Import {name} successfully.')
        except ModuleNotFoundError:
            if name in optional_package:
                print_color(f'{name} module is not found! Optionally install it.', bg=Color.CYAN.value)
            else:
                print_color(f'{name} module is not found! One should install it.', bg=Color.RED.value)
    return module

if __name__ == '__main__':
    module = check_module()
    print('Packages include:', module)