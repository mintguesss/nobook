"""把子行程綁進 Windows Job Object，父行程一死就一起收掉。

為什麼需要：`subprocess.terminate()` 只在父行程「有機會執行收尾」時有用。
主服務被工作管理員砍掉、被服務管理員強制停止、或當掉時，llama-server
會變成孤兒繼續佔著 VRAM 與 port——實測這會讓下一次量測全部失效
（詳見 README「量測時踩到的坑」第 1 點）。

Job Object 設了 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 之後，這件事由核心保證：
job handle 隨父行程結束而關閉，核心就把 job 裡的行程一起終止。

非 Windows 平台是 no-op。
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

log = logging.getLogger(__name__)

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

_job = None
_unavailable = False


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _ensure_job():
    """建立（或取回）本行程的 job，失敗回傳 None。"""
    global _job, _unavailable
    if _job is not None or _unavailable:
        return _job
    if sys.platform != "win32":
        _unavailable = True
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
                handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        _job = handle
        log.debug("已建立 Job Object，子行程將隨主行程一起結束")
    except Exception as e:      # 沒有權限等情況：退化為原本的 terminate 流程
        log.warning("無法建立 Job Object（%s）；"
                    "主服務若被強制終止，llama-server 可能變成孤兒", e)
        _unavailable = True
    return _job


def assign(pid: int) -> bool:
    """把 pid 綁進 job。成功回傳 True。"""
    job = _ensure_job()
    if job is None:
        return False
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not h:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not k32.AssignProcessToJobObject(job, h):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            k32.CloseHandle(h)
        return True
    except Exception as e:
        log.warning("無法把 pid %d 綁進 Job Object：%s", pid, e)
        return False
