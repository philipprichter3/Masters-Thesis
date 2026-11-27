
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from environments.classes.closed_schroedinger_env import CLosed_Schroedinger_Env
from environments.classes.lindblad_env import Lindblad_Environment
from environments.qm_env_wrapper import GymEnvironment
from logs.training_logger import TrainingLogger
from sac.sac import SAC

# ======================================================================
# ============================= PARAMETERS ============================
# ======================================================================

ENV_REGISTRY = {
    "lindblad": Lindblad_Environment,
    "closed_schroedinger": CLosed_Schroedinger_Env
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ======================================================================
# ============================= SAVING ============================
# ======================================================================

def save_experiment(
    agent: SAC,
    env: GymEnvironment,
    *,
    env_params: dict,
    hyper_params: dict,
    seed: int,
    steps: int,
    logger: TrainingLogger,
) -> Path:

    saved_at = datetime.now(UTC)
    timestamp = saved_at.strftime("%Y-%m-%dT%H-%M-%S-%fZ")

    directory = EXPERIMENTS_DIR / timestamp

    env_type = next(
        (
            name
            for name, cls in ENV_REGISTRY.items()
            if type(env.qm_env) is cls
        ),
        None,
    )

    metadata = {
        "saved_at_utc": saved_at.isoformat(),
        "env_type": env_type,
        "env_params": env_params,
        "hyper_params": hyper_params,
        "seed": int(seed),
        "steps": int(steps),
    }

    metadata_json = json.dumps(
        metadata,
        indent=2,
        allow_nan=False,
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    agent.save(directory / "agent.pt")

    logger.save(directory / "logger.pt")

    (directory / "metadata.json").write_text(
        metadata_json + "\n",
        encoding="utf-8",
    )

    print(f"Finished experiment saved: {directory.name}")

    return directory

# ======================================================================
# ============================= LOADING ============================
# ======================================================================

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


def load_experiment(experiment_id):

    directory = experiment_path(experiment_id)

    metadata = read_experiment_metadata(experiment_id)

    env_type = metadata["env_type"]

    env_class = ENV_REGISTRY[env_type]

    env = env_class(params=metadata["env_params"])

    gym_env = GymEnvironment(
        env,
        metadata["hyper_params"],
    )

    use_cuda = (
        bool(metadata["hyper_params"].get("USE_CUDA", False))
        and torch.cuda.is_available()
    )

    agent = SAC.load(
        directory / "agent.pt",
        gym_env.observation_space,
        gym_env.action_space,
        use_cuda=use_cuda,
        hidden_sizes=metadata["hyper_params"].get(
            "HIDDEN_SIZES", (512, 256)
        ),
    )

    logger_path = directory / "logger.pt"

    logger = (
        TrainingLogger.load(
            logger_path,
            metadata["hyper_params"],
        )
        if logger_path.exists()
        else TrainingLogger(metadata["hyper_params"])
    )

    return agent, gym_env, metadata, logger