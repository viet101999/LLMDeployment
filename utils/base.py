import asyncio
import contextlib
import logging
import threading
import time
import traceback
from asyncio import Lock
from typing import AsyncIterator

import aiohttp


class Base(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tasks = {}
        self.sessions = {}

    def add_task(self, task):
        task_id = id(task)

        def drop_task_callback(*args, **kwargs):
            self.tasks.pop(task_id, None)

        task.add_done_callback(drop_task_callback)
        self.tasks[task_id] = task

    @staticmethod
    def _current_thread_id():
        return threading.get_ident()

    @property
    def session(self):
        thread_id = self._current_thread_id()
        if not self.sessions.get(thread_id):
            self._init_session(force=True)
        return self.sessions[thread_id]

    @session.setter
    def session(self, _session):
        thread_id = self._current_thread_id()
        self.sessions[thread_id] = _session
        self.logger.info(f"Set new session {id(_session)} for thread {thread_id}")

    @staticmethod
    def generate_async_func(func, *args, **kwargs):
        async def _func():
            return await func(*args, **kwargs)

        return _func

    @staticmethod
    async def close_session(session):
        if session is not None:
            await asyncio.sleep(300)
            await session.close()

    def _init_session(self, force=False):
        thread_id = self._current_thread_id()
        old_session = self.sessions.get(thread_id)
        is_new_session = False
        if force or old_session is None:
            self.session = aiohttp.ClientSession()
            is_new_session = True

        if is_new_session and old_session is not None:
            asyncio.ensure_future(self.close_session(old_session))

    @contextlib.contextmanager
    def get_session(self):
        """
        Yield request session
        :return:
        """
        yield self.session

    async def call_back_func(self):
        await self.init_session(True)

    async def init_session(self, force=False):
        self._init_session(force=force)

    async def retry(self, func, n_retry: int = 5, **kwargs):
        for i in range(n_retry):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
            except Exception as e:
                error = traceback.format_exc()
                self.logger.error(f"Cannot make {func} due to error. \n"
                                  f"Error: {e}. \n"
                                  f"Detail: {error}")
                continue


class BaseWithMaxSessionsPerSecond(Base):
    def __init__(self, max_sessions_per_second: int = 5):
        super(BaseWithMaxSessionsPerSecond, self).__init__()
        self.time_frames = {}
        self.max_sessions_per_second = max_sessions_per_second
        self.lock = Lock()

    @staticmethod
    def _current_time_id():
        return int(time.time())

    def get_frame_session_by_id(self, frame_id: int):
        return self.time_frames.get(frame_id, [])

    def add_frame_session(self) -> int:
        """
        Return current frame Id
        :return:
        """
        current_frame_id = self._current_time_id()
        if current_frame_id not in self.time_frames:
            self.time_frames[current_frame_id] = []
        self.time_frames[current_frame_id].append(1)
        return current_frame_id

    def get_rate_by_id(self, frame_id: int):
        """
        Get rate at frame_id
        :param frame_id:
        :return:
        """
        current_frame_sessions = self.get_frame_session_by_id(frame_id)
        return len(current_frame_sessions)

    @property
    def rate(self):
        """
        Get current rate by current frame id
        :return:
        """
        current_frame_id = self._current_time_id()
        return self.get_rate_by_id(current_frame_id)

    @property
    def rate_info(self):
        """
        Get current frame rate as JSON format
        :return:
        """
        current_frame_id = self._current_time_id()
        return {
            "time_frame": current_frame_id,
            "rate": self.get_rate_by_id(current_frame_id)
        }

    @property
    def full(self):
        return self.rate >= self.max_sessions_per_second

    async def get_slot(self):
        """
        Wait until current frame is not full
        :return:
        """
        while self.full:
            await asyncio.sleep(0.1)
        return self.add_frame_session()

    async def close_session(self, session):
        """
        Async Close a session after 300s and clear all old frames
        :param session:
        :return:
        """
        if session is not None:
            await asyncio.sleep(300)
            await session.close()
            current_frame_id = self._current_time_id()
            for time_frame, _ in list(self.time_frames.items()):
                if current_frame_id > time_frame:
                    self.time_frames.pop(time_frame, None)

    @contextlib.asynccontextmanager
    async def get_session(self) -> AsyncIterator[aiohttp.ClientSession]:
        """
        Yield request Session
        :return:
        """
        await self.lock.acquire()
        await self.get_slot()
        self.logger.info(f"Current Rate: {self.rate_info}")
        self.lock.release()
        yield self.session
