"""Tests for ``scripts/launch_dlc_finetune_ec2.py`` — SA-finetune passthrough.

Pure user-data string assembly + boto3 call assertions. No real AWS
calls. Mirrors the ``_FakeS3`` pattern from
``tests/scripts/test_declare_dlc_champion.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

# The launcher imports ec2_constants and ec2_utils with side-effecty
# defaults. Patch those for safe import in the test environment.
with patch.dict(
    sys.modules,
    {
        # Ensure boto3 is available for module-level imports inside the script.
    },
):
    import launch_dlc_finetune_ec2 as launcher  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_credentials(monkeypatch):
    """Prevent get_s3_credentials from reaching the host's AWS profile."""
    monkeypatch.setattr(
        launcher,
        "get_s3_credentials",
        lambda: ("AKIA-TEST", "SECRET-TEST", "ap-southeast-2"),
    )


# ---------------------------------------------------------------------------
# argparse + epochs resolution
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_sa_finetune_parses(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args(["--sa-finetune"])
        assert args.sa_finetune is True

    def test_default_off(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args([])
        assert args.sa_finetune is False

    def test_compatible_with_infer_only(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args(["--sa-finetune", "--infer-only"])
        assert args.sa_finetune is True
        assert args.infer_only is True

    def test_compatible_with_dry_run(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args(["--sa-finetune", "--dry-run"])
        assert args.sa_finetune is True
        assert args.dry_run is True

    def test_epochs_default_none(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args([])
        assert args.epochs is None

    def test_explicit_epochs_honoured(self):
        parser = launcher._build_arg_parser()
        args = parser.parse_args(["--sa-finetune", "--epochs", "150"])
        assert args.epochs == 150


class TestResolveEpochs:
    def test_sa_default(self):
        assert launcher._resolve_epochs(None, sa_finetune=True) == 200

    def test_imagenet_default(self):
        assert launcher._resolve_epochs(None, sa_finetune=False) == 400

    def test_explicit_overrides(self):
        assert launcher._resolve_epochs(75, sa_finetune=True) == 75
        assert launcher._resolve_epochs(75, sa_finetune=False) == 75


# ---------------------------------------------------------------------------
# build_user_data — the core string-assembly logic
# ---------------------------------------------------------------------------


class TestBuildUserData:
    def test_sa_finetune_appended_to_run_dlc_retrain_invocation(self):
        ud = launcher.build_user_data(epochs=120, sa_finetune=True)
        assert "--sa-finetune" in ud
        # The flag appears on the python3 invocation line.
        for line in ud.splitlines():
            if "scripts/run_dlc_retrain.py" in line and "python3" in line:
                assert "--sa-finetune" in line
                return
        pytest.fail("run_dlc_retrain.py invocation line not found in user-data")

    def test_sa_finetune_off_omits_flag(self):
        ud = launcher.build_user_data(epochs=400, sa_finetune=False)
        assert "--sa-finetune" not in ud

    def test_sa_finetune_compatible_with_infer_only(self):
        ud = launcher.build_user_data(infer_only=True, sa_finetune=True)
        assert "--sa-finetune" in ud
        assert "--infer-only" in ud

    def test_sa_finetune_label_in_log(self):
        ud = launcher.build_user_data(epochs=120, sa_finetune=True)
        # The visible label gets the [SA fine-tune] suffix.
        assert "SA fine-tune" in ud

    def test_default_user_data_is_train_plus_infer(self):
        ud = launcher.build_user_data(epochs=400)
        assert "--epochs 400" in ud
        # Mode label includes "train + inference" when neither flag is set.
        assert "train + inference" in ud


# ---------------------------------------------------------------------------
# launch — boto3 mocked at the boto3.client(...) level
# ---------------------------------------------------------------------------


def _make_mock_clients() -> tuple[MagicMock, MagicMock]:
    ec2 = MagicMock()
    s3 = MagicMock()
    s3.head_object.return_value = {}
    s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "dlc-retrain/models/snapshot-best-100.pt"}],
    }
    # run_instances response shape.
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    # Waiter + describe_instances for IP printout.
    waiter = MagicMock()
    ec2.get_waiter.return_value = waiter
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [{"PublicIpAddress": "1.2.3.4"}],
            }
        ],
    }
    return ec2, s3


def _patched_clients(monkeypatch, ec2: MagicMock, s3: MagicMock) -> None:
    """Patch boto3.client so launcher.launch picks up our mocks."""

    def _client(name: str, region_name: str = ""):
        return ec2 if name == "ec2" else s3

    monkeypatch.setattr(launcher.boto3, "client", _client)


class TestLaunchInstance:
    def test_disk_size_120gb_under_sa_finetune(self, monkeypatch):
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        launcher.launch(maxiters=0, epochs=120, sa_finetune=True)
        ec2.run_instances.assert_called_once()
        kwargs = ec2.run_instances.call_args.kwargs
        bdm = kwargs["BlockDeviceMappings"]
        assert bdm[0]["Ebs"]["VolumeSize"] == 120
        # Instance type unchanged.
        assert kwargs["InstanceType"] == "g4dn.2xlarge"

    def test_disk_size_100gb_default(self, monkeypatch):
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        launcher.launch(maxiters=0, epochs=400, sa_finetune=False)
        ec2.run_instances.assert_called_once()
        kwargs = ec2.run_instances.call_args.kwargs
        assert kwargs["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 100

    def test_dry_run_does_not_launch(self, monkeypatch, capsys):
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        launcher.launch(maxiters=0, sa_finetune=True, dry_run=True)
        ec2.run_instances.assert_not_called()
        out = capsys.readouterr().out
        # Dry-run prints the user-data; the SA flag should be visible.
        assert "--sa-finetune" in out

    def test_user_data_contains_sa_flag(self, monkeypatch):
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        launcher.launch(maxiters=0, sa_finetune=True)
        kwargs = ec2.run_instances.call_args.kwargs
        assert "--sa-finetune" in kwargs["UserData"]

    def test_instance_type_unchanged_under_sa(self, monkeypatch):
        """g4dn.2xlarge stays the default per architect open-question #1."""
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        launcher.launch(maxiters=0, sa_finetune=True)
        kwargs = ec2.run_instances.call_args.kwargs
        assert kwargs["InstanceType"] == "g4dn.2xlarge"

    def test_module_constants(self):
        # Confirm INSTANCE_TYPE constant unchanged.
        assert launcher.INSTANCE_TYPE == "g4dn.2xlarge"


# ---------------------------------------------------------------------------
# status / progress / terminate short-circuit launch
# ---------------------------------------------------------------------------


class TestShortCircuits:
    def test_status_calls_describe_not_run(self, monkeypatch):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {"Reservations": []}
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": ec2)
        launcher.status()
        ec2.describe_instances.assert_called_once()
        ec2.run_instances.assert_not_called()

    def test_main_dispatches_status(self, monkeypatch):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {"Reservations": []}
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": ec2)
        monkeypatch.setattr(sys, "argv", ["launch_dlc_finetune_ec2.py", "--status"])
        launcher.main()
        ec2.describe_instances.assert_called_once()
        ec2.run_instances.assert_not_called()

    def test_main_dispatches_terminate(self, monkeypatch):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {"Reservations": []}
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": ec2)
        monkeypatch.setattr(sys, "argv", ["launch_dlc_finetune_ec2.py", "--terminate"])
        launcher.main()
        ec2.describe_instances.assert_called_once()

    def test_main_mutual_exclusion(self, monkeypatch, capsys):
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": MagicMock())
        monkeypatch.setattr(
            sys, "argv",
            ["launch_dlc_finetune_ec2.py", "--infer-only", "--train-only"],
        )
        with pytest.raises(SystemExit):
            launcher.main()
        err = capsys.readouterr().out
        assert "mutually exclusive" in err

    def test_main_dispatches_launch_under_sa_finetune(self, monkeypatch):
        ec2, s3 = _make_mock_clients()
        _patched_clients(monkeypatch, ec2, s3)
        monkeypatch.setattr(
            sys, "argv",
            ["launch_dlc_finetune_ec2.py", "--sa-finetune"],
        )
        launcher.main()
        ec2.run_instances.assert_called_once()
        kwargs = ec2.run_instances.call_args.kwargs
        # SA path -> EBS 120, user-data carries flag.
        assert kwargs["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 120
        assert "--sa-finetune" in kwargs["UserData"]

    def test_progress_prints(self, monkeypatch):
        s3 = MagicMock()
        # First get_object returns valid JSON; second raises.
        body_json = MagicMock()
        body_json.read.return_value = b'{"status":"running"}'
        body_csv = MagicMock()
        body_csv.read.return_value = b"timestamp,gpu\nt1,75 %\nt2,80 %\n"
        s3.get_object.side_effect = [
            {"Body": body_json},
            {"Body": body_csv},
        ]
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": s3)
        launcher.progress()
        # Two get_object calls (progress JSON + GPU CSV).
        assert s3.get_object.call_count == 2

    def test_progress_handles_missing_files(self, monkeypatch, capsys):
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("nope")
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": s3)
        launcher.progress()
        out = capsys.readouterr().out
        assert "No progress data yet" in out

    def test_launch_aborts_when_no_labels_on_s3(self, monkeypatch, capsys):
        ec2 = MagicMock()
        s3 = MagicMock()
        s3.head_object.side_effect = Exception("missing")
        _patched_clients(monkeypatch, ec2, s3)
        with pytest.raises(SystemExit):
            launcher.launch(maxiters=0, sa_finetune=False)
        out = capsys.readouterr().out
        assert "no labeled data" in out

    def test_launch_infer_only_aborts_without_models(self, monkeypatch, capsys):
        ec2 = MagicMock()
        s3 = MagicMock()
        s3.head_object.return_value = {}
        s3.list_objects_v2.return_value = {"Contents": []}
        _patched_clients(monkeypatch, ec2, s3)
        with pytest.raises(SystemExit):
            launcher.launch(maxiters=0, infer_only=True, sa_finetune=False)
        out = capsys.readouterr().out
        assert "model weights" in out.lower()
        assert "none were found" in out.lower()

    def test_terminate_calls_terminate_not_run(self, monkeypatch):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-x", "State": {"Name": "running"}}]}]
        }
        monkeypatch.setattr(launcher.boto3, "client", lambda n, region_name="": ec2)
        launcher.terminate()
        ec2.terminate_instances.assert_called_once()
        ec2.run_instances.assert_not_called()
