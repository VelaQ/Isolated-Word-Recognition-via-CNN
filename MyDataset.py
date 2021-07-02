import os
from typing import Tuple, Optional, Union
from pathlib import Path

from torchaudio import load as AudioLoad
from torch.utils.data import Dataset
from torch import Tensor


FOLDER_IN_ARCHIVE = "MyDataset"
EXCEPT_FOLDER = "_background_noise_"

# 加载列表
def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output

# 加载语音
def load_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    # Load audio
    waveform, sample_rate = AudioLoad(filepath)
    return waveform, sample_rate, label

# 定义数据集类
class MyDataset(Dataset):

    # 初始化方法
    def __init__(self,
                 root: Union[str, Path],
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 subset: Optional[str] = None,
                 ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        root = os.fspath(root)

        self._path = os.path.join(root, folder_in_archive)


        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [
                w for w in walker
                if EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [w for w in walker if EXCEPT_FOLDER not in w]

    # 定义得到样本方法
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_item(fileid, self._path)


    def __len__(self) -> int:
        return len(self._walker)
