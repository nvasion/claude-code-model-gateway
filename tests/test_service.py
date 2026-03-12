"""Tests for the service module (daemon entry point)."""

import os
import signal
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from src.service import ServiceManager, _env, _env_int, main


class TestServiceManager:
    """Tests for the ServiceManager class."""

    def test_init_defaults(self):
        """ServiceManager initialises with sensible defaults."""
        mgr = ServiceManager()
        assert mgr.host == "0.0.0.0"
        assert mgr.port == 8080
        assert mgr.timeout == 300
        assert mgr.log_level == "info"
        assert mgr.log_format == "json"
        assert mgr.log_file is None
        assert mgr.pid_file is None
        assert mgr.config_file is None

    def test_init_custom(self):
        """ServiceManager stores custom configuration."""
        mgr = ServiceManager(
            host="127.0.0.1",
            port=9090,
            timeout=60,
            log_level="debug",
            log_format="standard",
            log_file="/tmp/test.log",
            pid_file="/tmp/test.pid",
            config_file="/tmp/gateway.yaml",
        )
        assert mgr.host == "127.0.0.1"
        assert mgr.port == 9090
        assert mgr.timeout == 60
        assert mgr.log_level == "debug"
        assert mgr.log_format == "standard"
        assert mgr.log_file == "/tmp/test.log"
        assert mgr.pid_file == "/tmp/test.pid"
        assert mgr.config_file == "/tmp/gateway.yaml"

    def test_write_pid_file(self, tmp_path):
        """PID file is written correctly."""
        pid_file = str(tmp_path / "test.pid")
        mgr = ServiceManager(pid_file=pid_file)
        mgr._write_pid_file()

        assert Path(pid_file).exists()
        content = Path(pid_file).read_text().strip()
        assert content == str(os.getpid())

    def test_remove_pid_file(self, tmp_path):
        """PID file is removed on cleanup."""
        pid_file = str(tmp_path / "test.pid")
        Path(pid_file).write_text("12345")
        mgr = ServiceManager(pid_file=pid_file)
        mgr._remove_pid_file()
        assert not Path(pid_file).exists()

    def test_remove_pid_file_missing(self, tmp_path):
        """Removing a non-existent PID file does not raise."""
        pid_file = str(tmp_path / "nonexistent.pid")
        mgr = ServiceManager(pid_file=pid_file)
        mgr._remove_pid_file()  # Should not raise

    def test_write_pid_file_no_pidfile(self):
        """No PID file is written when pid_file is None."""
        mgr = ServiceManager(pid_file=None)
        mgr._write_pid_file()  # Should be a no-op

    def test_load_env_file(self, tmp_path):
        """Environment file is parsed and loaded correctly."""
        env_file = tmp_path / "env"
        env_file.write_text(
            "# comment\n"
            "FOO=bar\n"
            "  BAZ = qux  \n"
            'QUOTED="hello world"\n'
            "\n"
            "EMPTY=\n"
        )
        ServiceManager._load_env_file(str(env_file))

        assert os.environ.get("FOO") == "bar"
        assert os.environ.get("BAZ") == "qux"
        assert os.environ.get("QUOTED") == "hello world"

        # Clean up
        for key in ("FOO", "BAZ", "QUOTED", "EMPTY"):
            os.environ.pop(key, None)

    def test_load_env_file_missing(self):
        """Loading a missing env file does not raise."""
        ServiceManager._load_env_file("/nonexistent/env")

    def test_shutdown_event(self):
        """Shutdown event is set by the signal handler."""
        mgr = ServiceManager()
        assert not mgr._shutdown_event.is_set()
        mgr._handle_shutdown(signal.SIGTERM, None)
        assert mgr._shutdown_event.is_set()

    def test_handle_reload(self, tmp_path):
        """SIGHUP handler reloads environment."""
        env_file = tmp_path / "env"
        env_file.write_text("RELOAD_TEST_VAR=reloaded\n")

        mgr = ServiceManager()
        with mock.patch.dict(os.environ, {"GATEWAY_ENV_FILE": str(env_file)}):
            mgr._handle_reload(signal.SIGHUP, None)
            assert os.environ.get("RELOAD_TEST_VAR") == "reloaded"
        os.environ.pop("RELOAD_TEST_VAR", None)


class TestEnvHelpers:
    """Tests for the _env and _env_int helper functions."""

    def test_env_returns_value(self):
        """_env returns the environment variable value."""
        with mock.patch.dict(os.environ, {"TEST_ENV_VAR": "hello"}):
            assert _env("TEST_ENV_VAR", "default") == "hello"

    def test_env_returns_default(self):
        """_env returns the default when variable is not set."""
        os.environ.pop("MISSING_VAR", None)
        assert _env("MISSING_VAR", "fallback") == "fallback"

    def test_env_int_returns_value(self):
        """_env_int returns the parsed integer."""
        with mock.patch.dict(os.environ, {"TEST_INT_VAR": "42"}):
            assert _env_int("TEST_INT_VAR", 0) == 42

    def test_env_int_returns_default(self):
        """_env_int returns the default when variable is not set."""
        os.environ.pop("MISSING_INT_VAR", None)
        assert _env_int("MISSING_INT_VAR", 99) == 99

    def test_env_int_invalid_value(self):
        """_env_int returns default for non-integer values."""
        with mock.patch.dict(os.environ, {"BAD_INT": "not_a_number"}):
            assert _env_int("BAD_INT", 10) == 10


class TestServiceRun:
    """Integration tests for the ServiceManager.run method."""

    def test_run_starts_and_stops(self):
        """Service starts and can be shut down via the shutdown event."""
        mgr = ServiceManager(
            host="127.0.0.1",
            port=0,  # Will fail to bind on port 0 in some cases
            timeout=5,
            log_level="warning",
            log_format="standard",
        )

        # We'll simulate a quick shutdown
        def delayed_shutdown():
            time.sleep(0.3)
            mgr._shutdown_event.set()
            if mgr._server:
                mgr._server.shutdown()

        shutdown_thread = threading.Thread(target=delayed_shutdown, daemon=True)
        shutdown_thread.start()

        exit_code = mgr.run()
        assert exit_code == 0

    def test_main_entry_reads_env(self):
        """The main() function reads from environment variables."""
        env = {
            "GATEWAY_HOST": "127.0.0.1",
            "GATEWAY_PORT": "0",
            "GATEWAY_TIMEOUT": "5",
            "GATEWAY_LOG_LEVEL": "warning",
            "GATEWAY_LOG_FORMAT": "standard",
            "GATEWAY_LOG_FILE": "",
            "GATEWAY_PID_FILE": "",
            "GATEWAY_CONFIG": "",
        }

        with mock.patch.dict(os.environ, env):
            with mock.patch("src.service.ServiceManager.run", return_value=0):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
