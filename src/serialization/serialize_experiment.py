
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from environments.qm_env_wrapper import GymEnvironment
from sac.sac import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def save_experiment(
        agent: SAC,
        *,
        env_params: dict,
        hyper_params: dict,
        seed: int,
        steps: int,
    ) -> Path:

    saved_at = datetime.now(UTC)

    timestamp = saved_at.strftime("%Y-%m-%dT%H-%M-%S-%fZ")

    directory = EXPERIMENTS_DIR / timestamp

    metadata = {
        "saved_at_utc": saved_at.isoformat(),

        "env_params": env_params,
        "hyper_params": hyper_params,

        "seed": int(seed),
        "steps": int(steps),
    }

    metadata_json = json.dumps(
        metadata, 
        indent=2, 
        allow_nan=False
    )

    directory.mkdir(
        parents=True, 
        exist_ok=False
    )

    agent.save(directory / "agent.pt")

    (directory / "metadata.json").write_text(
        metadata_json + "\n", encoding="utf-8"
    )

    print(f"Finished experiment saved: {directory.name}")

    return directory


def experiment_path(experiment_id):

    return EXPERIMENTS_DIR / experiment_id


def read_experiment_metadata(experiment_id) -> dict:

    directory = experiment_path(experiment_id)

    metadata_path = Path(directory) / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if "HIDDEN_SIZES" in metadata["hyper_params"]:

        metadata["hyper_params"]["HIDDEN_SIZES"] = tuple(
            metadata["hyper_params"]["HIDDEN_SIZES"]
        )

    return metadata


def load_experiment(experiment_id, env: GymEnvironment):

    directory = experiment_path(experiment_id)

    metadata = read_experiment_metadata(experiment_id)

    use_cuda = (
        bool(metadata["hyper_params"].get("USE_CUDA", False))
        and torch.cuda.is_available()
    )

    agent = SAC.load(
        Path(directory) / "agent.pt",
        env.observation_space,
        env.action_space,
        use_cuda=use_cuda,
        hidden_sizes=metadata["hyper_params"].get("HIDDEN_SIZES", (512,256))
    )
    
    return agent, metadata