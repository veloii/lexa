import json
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

from .constants import METADATA_FILENAME
from .export_fmadapter import MetadataKeys
from .export_utils import camelize

logger = logging.getLogger(__name__)


class AssetPackBuilder:
    """
    A wrapper for the Cappuccino packaging tool.

    Providing an interface for constructing and executing the packaging command.
    """

    class Platforms(Enum):
        """
        Platforms to download the adapter asset.
        """

        iOS = "iOS"
        macOS = "macOS"
        visionOS = "visionOS"

    class DownloadPolicy(Enum):
        """
        Policy for download the adapter asset.

            Essential: Blocks app launch until it is fully downloaded.
            Pre-Fetch: Downloaded automatically after install but you can still launch the app.
            On Demand: Triggered on demand by developers via public API
        """

        ESSENTIAL = "essential"
        PREFETCH = "prefetch"
        ON_DEMAND = "onDemand"

    class InstallationEventType(Enum):
        """
        The event type for asset downloading.
        Relevant for asset packs with essential or prefetch download policies.

            First install: asset pack be delivered automatically only to new users
            Update: only to existing users who update to a new version of the app
            Both: To both groups first install or update
        """

        FIRST_INSTALLTION = "firstInstallation"
        SUBSEQUENT_UPDATE = "subsequentUpdate"

    def __init__(
        self,
        fmadapter_path: str | Path,
        output_path: str | Path = ".",
        platforms: list | None = None,
        download_policy: str | None = None,
        installation_event_type: str | None = None,
        quiet_mode: bool = False,
    ) -> "AssetPackBuilder":
        """
        Construct a AssetPackBuilder object.

        Args:
            fmadapter_path (str | Path): The path to fmadapter
            output_path (str | Path): The output path where the asset pack will be built.
            platforms (str | None): The platforms to which the asset pack download is compatible with.
            download_policy (str | None): The policy for download the adapter asset.
            installation_event_type (str | None): The event type for asset downloading.
            quiet_mode (bool): The flag for suppressing messages.
        """
        self.fmadapter_path = Path(fmadapter_path)
        self.asset_pack_id = self._extract_adapter_identifier()
        self.output_asset_pack_path = self._get_output_asset_pack_path(output_path)
        self.platforms = (
            platforms
            if platforms is not None
            else " ".join([platform.value for platform in list(AssetPackBuilder.Platforms)])
        )
        self.download_policy = download_policy or "prefetch"
        self.installation_event_type = installation_event_type or "firstInstallation"
        self.quiet_mode = quiet_mode

    def _extract_adapter_identifier(self) -> str:
        metadata_path = self.fmadapter_path / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"{METADATA_FILENAME} not found at {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        adapter_identifier = camelize(MetadataKeys.ADAPTER_IDENTIFIER.value)
        if adapter_identifier not in metadata:
            err = f"Missing {adapter_identifier} in metadata."
            raise KeyError(err)
        return metadata[adapter_identifier]

    def _get_output_asset_pack_path(self, output_path: str | Path) -> Path:
        """
        Convert user provided output_path to a Path for building the asset_pack.

        Args:
            output_path (str | Path): The user provided output path for the asset_pack.
        """
        output_path = Path(output_path)
        fmadapter_name = Path(self.fmadapter_path.name).stem

        if str(output_path).endswith(".aar"):
            return Path(output_path).resolve(strict=False)

        if output_path.exists() and output_path.is_dir():
            return (output_path / f"{fmadapter_name}.aar").resolve(strict=False)

        if output_path.parent.exists() and output_path.parent.is_dir():
            return (output_path.parent / f"{output_path.name}.aar").resolve(strict=False)
        else:
            return output_path.resolve(strict=False)

    def __call__(self) -> str:
        command = ["xcrun", "ba-package", "foundation-models", "package"]

        if self.output_asset_pack_path.exists():
            raise FileExistsError("Asset pack already exists at: %s", self.output_asset_pack_path)

        current_directory = Path(os.curdir).resolve()
        fmadapter_directory, fmadapter_file = self.fmadapter_path.parent, self.fmadapter_path.name
        if fmadapter_directory:
            try:
                os.chdir(fmadapter_directory)
            except OSError as e:
                raise RuntimeError(f"Failed to change directory to the fmadapter parent directory: {fmadapter_directory} with error: {e}.")

        # Required arguments
        command.extend(
            [
                "--adapter-path",
                fmadapter_file,
                "--asset-pack-id",
                self.asset_pack_id,
                "--output-path",
                str(self.output_asset_pack_path),
            ]
        )

        # Optional arguments
        if self.platforms:
            platforms = [platform.value for platform in self.platforms]
            command.extend(["--platforms", *platforms])

        if self.download_policy:
            command.append("--" + self.download_policy.value)

        if self.installation_event_type:
            event_types = [self.installation_event_type.value]
            command.extend(["--installation-event-types", *event_types])

        if self.quiet_mode:
            command.append("--quiet")

        subprocess.run(command, check=False)

        os.chdir(current_directory)

        return str(self.output_asset_pack_path)

def produce_asset_pack(
    fmadapter_path: str | Path,
    output_path: str | Path = ".",
    platforms: list | None = None,
    download_policy: str | None = None,
    installation_event_type: str | None = None,
    quiet_mode: bool = False
) -> str:
    """
    Produce asset pack for distribution using the .fmadapter

    Args:
        fmadapter_path (str | Path): The path to fmadapter
        output_path (str | Path): The output path where the asset pack will be built.
        platforms (str | None): The platforms to which the asset pack download is compatible with.
        download_policy (str | None): The policy for download the adapter asset.
        installation_event_type (str | None): The event type for asset downloading.
        quiet_mode (bool): The flag for suppressing messages.
    """
    asset_pack_builder = AssetPackBuilder(
        fmadapter_path=fmadapter_path,
        output_path=output_path,
        platforms=platforms,
        download_policy=download_policy,
        installation_event_type=installation_event_type,
        quiet_mode=quiet_mode
    )
    return asset_pack_builder()
