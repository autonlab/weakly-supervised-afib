import pickle
from pathlib import Path
from typing import Union

def saveTo(objToStore: object, dst: Union[Path, str], writetype='wb+'):
    if 'b' in writetype:
        with open(dst, 'wb+') as writefile:
            pickle.dump(objToStore, writefile)
    else:
        with open(dst, 'w+') as writefile:
            writefile.write(objToStore)
