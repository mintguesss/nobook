"""把整堂課的原始音訊存成 FLAC（規格 §10 的選用開關）。

規格說原始音訊預設不保存，理由是「三小時 16kHz PCM 約 350MB，且已無用途」。
但實際用途出現了：使用者要能回頭聽某一段確認 ASR 有沒有聽錯，
而且未來換更好的模型可以重跑。FLAC 無損壓縮後約 100MB/堂。

邊收邊寫，不在記憶體裡累積——三小時 float32 會吃掉 700MB RAM。
"""
from __future__ import annotations

import datetime
import logging
import re
import threading
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

# Windows 不允許的字元，加上空白（檔名帶空白在指令列很難用）
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def build_filename(course, started_at, session_id, directory=None,
                   suffix="") -> Path:
    """課名_日期_時間.flac。

    用 uuid 當檔名的話，要從檔案總管找某一堂的錄音得先去查資料庫。
    課名加日期時間才看得出是哪一堂——同一天同一門課會有好幾段，
    所以時間要留到分鐘。

    任何一項缺漏就退回 uuid：檔名只是給人看的，不值得為了好看讓錄音存不下來。
    """
    d = Path(directory or config.AUDIO_DIR)
    name = _BAD_CHARS.sub("", str(getattr(course, "name", "") or ""))
    try:
        ts = datetime.datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        ts = None
    if not name or ts is None:
        return d / ("%s.flac" % session_id)

    stem = "%s_%s_%s" % (name, ts.strftime("%Y-%m-%d"), ts.strftime("%H%M"))
    if suffix:
        stem += "_" + suffix
    path = d / ("%s.flac" % stem)
    # 同一分鐘內開兩段（接續錄音、或按錯重開）會撞名
    n = 2
    while path.exists():
        path = d / ("%s_%d.flac" % (stem, n))
        n += 1
    return path


class AudioRecorder:
    """單一 session 的音訊寫入器。write() 可從任意執行緒呼叫。"""

    def __init__(self, session_id: str, directory=None,
                 course=None, started_at=None):
        self.session_id = session_id
        self.dir = Path(directory or config.AUDIO_DIR)
        if course is None and started_at is None:
            self.path = self.dir / ("%s.flac" % session_id)
        else:
            self.path = build_filename(course, started_at, session_id,
                                       self.dir)
        self._f = None
        self._lock = threading.Lock()
        self._samples = 0
        self._failed = False

    @property
    def seconds(self) -> float:
        return self._samples / float(config.SAMPLE_RATE)

    def _ensure(self):
        if self._f is not None or self._failed:
            return self._f
        try:
            import soundfile as sf
            self.dir.mkdir(parents=True, exist_ok=True)
            self._f = sf.SoundFile(str(self.path), mode="w",
                                   samplerate=config.SAMPLE_RATE,
                                   channels=1, format="FLAC", subtype="PCM_16")
        except Exception as e:
            # 存檔失敗絕不能影響逐字稿（規格 §7.4 的優先級）
            log.error("無法建立音訊檔 %s：%s（錄音保存停用，逐字稿續行）",
                      self.path, e)
            self._failed = True
        return self._f

    def write(self, pcm: np.ndarray) -> None:
        with self._lock:
            f = self._ensure()
            if f is None:
                return
            try:
                f.write(pcm)
                self._samples += len(pcm)
            except Exception as e:
                log.error("寫入音訊失敗：%s（停用保存，逐字稿續行）", e)
                self._failed = True
                self._close_locked()

    def _close_locked(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def close(self):
        with self._lock:
            self._close_locked()
        if self._failed or not self.path.exists():
            return None
        size = self.path.stat().st_size
        log.info("音訊已保存 %s（%.1f 分鐘，%.0f MB）",
                 self.path.name, self.seconds / 60, size / 1e6)
        return str(self.path)
