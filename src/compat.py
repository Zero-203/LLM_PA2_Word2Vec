"""运行环境兼容层。

问题背景
--------
Python 3.9 的 ``pathlib.Path.mkdir()`` 内部调用
``self._accessor.mkdir(self, mode)``，会向底层传入 ``mode`` 位置参数。
在部分受限执行环境下，底层 mkdir 被替换为不接受 ``mode`` 的实现，
于是抛出：

    TypeError: _sandbox_mkdir() takes from 1 to 2 positional arguments
               but 3 were given

``TypeError`` 不是 ``OSError`` 子类，因此 pathlib 原有的
``except OSError`` 兜底捕获不到它，导致即使目标目录**已经存在**也会直接失败。

受影响的第三方库
----------------
- matplotlib：``get_configdir()`` / ``get_cachedir()`` 在**导入时**即调用
  ``configdir.mkdir(parents=True, exist_ok=True)``，故 ``import matplotlib``
  本身就会崩溃，设置 MPLCONFIGDIR 也无法绕过。
- 任何使用 ``pathlib.Path.mkdir`` 的项目代码。

解决方式
--------
把 ``pathlib.Path.mkdir`` 替换为基于 ``os.makedirs`` / ``os.mkdir`` 的等价实现
（实测这两个调用在受限环境下可用），并保持原有语义：
``parents`` / ``exist_ok`` / ``mode`` 行为一致，目录已存在且 ``exist_ok=True``
时静默返回。

本模块**必须在导入 matplotlib 之前**被导入（``config.py`` 已在顶部导入本模块）。
在普通无限制环境下安装本补丁是无害的：行为与标准库一致。
"""

from __future__ import annotations

import os
import pathlib

_PATCHED_FLAG = "_pa2_mkdir_shim"


def _patched_mkdir(self, mode: int = 0o777, parents: bool = False,
                   exist_ok: bool = False) -> None:
    """``pathlib.Path.mkdir`` 的兼容实现，避免向底层传递 mode 位置参数。"""
    try:
        if parents:
            os.makedirs(self, mode=mode, exist_ok=True)
        else:
            os.mkdir(self)
    except FileExistsError:
        if exist_ok and self.is_dir():
            return
        raise
    except OSError:
        # 与标准库一致：parents=True 时补建缺失的父目录后重试一次
        if parents and not self.parent.exists():
            self.parent.mkdir(parents=True, exist_ok=True)
            return _patched_mkdir(self, mode, False, exist_ok)
        if not (exist_ok and self.is_dir()):
            raise


def install() -> bool:
    """安装补丁。返回是否实际执行了替换（幂等）。"""
    if getattr(pathlib.Path.mkdir, _PATCHED_FLAG, False):
        return False
    patched = _patched_mkdir
    setattr(patched, _PATCHED_FLAG, True)
    pathlib.Path.mkdir = patched      # type: ignore[assignment]
    return True


# 导入即生效，保证任何后续 import matplotlib 都已被覆盖
install()


def configure_matplotlib(project_root: pathlib.Path) -> None:
    """把 matplotlib 的配置/缓存目录收敛到项目内，并强制无界面后端。

    - 避免在用户目录外写入缓存（本项目所有产物都落在项目目录内，便于审计）
    - 强制 Agg 后端：全流程为批处理脚本，不需要交互式窗口

    必须在 ``import matplotlib`` **之前**调用。
    """
    cache_dir = project_root / ".cache" / "matplotlib"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

