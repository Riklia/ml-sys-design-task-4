import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.cloud import aiplatform
from google.cloud import storage


PROJECT_ID = os.getenv("PROJECT_ID", "project-842f9db3-2713-4273-820")
LOCATION = os.getenv("LOCATION", "us-central1")
BUCKET_URI = os.getenv("BUCKET_URI", "gs://ml-sys-design-labs")
DISPLAY_NAME = os.getenv("DISPLAY_NAME", "mnist-hpt")
MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "10"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "2"))

TRAINING_IMAGE = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest"
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-4")


def run(command):
    subprocess.run(command, check=True)


def parse_bucket_uri(bucket_uri):
    bucket_path = bucket_uri.replace("gs://", "").rstrip("/")
    parts = bucket_path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def upload_file(local_path, bucket_uri, object_name):
    bucket_name, prefix = parse_bucket_uri(bucket_uri)
    object_path = f"{prefix}/{object_name}".strip("/")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_filename(str(local_path))

    return f"gs://{bucket_name}/{object_path}"


def build_package():
    if not Path("setup.py").exists():
        raise RuntimeError("Run this script from the folder that contains setup.py.")

    if not Path("trainer/__init__.py").exists() or not Path("trainer/task.py").exists():
        raise RuntimeError("Expected trainer/__init__.py and trainer/task.py.")

    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("build", ignore_errors=True)

    run([sys.executable, "-m", "pip", "install", "-q", "build"])
    run([sys.executable, "-m", "build", "--sdist"])

    packages = list(Path("dist").glob("*.tar.gz"))

    if not packages:
        raise RuntimeError("Package was not created.")

    return packages[0]


def submit_job():
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_URI,
    )

    package_path = build_package()

    package_uri = upload_file(
        package_path,
        BUCKET_URI,
        f"packages/{DISPLAY_NAME}/{package_path.name}",
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": MACHINE_TYPE,
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": TRAINING_IMAGE,
                "package_uris": [package_uri],
                "python_module": "trainer.task",
                "args": [
                    "--epochs=4",
                ],
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name=f"{DISPLAY_NAME}-trial",
        worker_pool_specs=worker_pool_specs,
    )

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=DISPLAY_NAME,
        custom_job=custom_job,
        metric_spec={
            "val_accuracy": "maximize",
        },
        parameter_spec={
            "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                min=0.0001,
                max=0.01,
                scale="log",
            ),
            "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
                values=[32, 64, 128],
                scale="linear",
            ),
            "hidden_units": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
                values=[64, 128, 256],
                scale="linear",
            ),
            "dropout_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                min=0.1,
                max=0.5,
                scale="linear",
            ),
        },
        max_trial_count=MAX_TRIAL_COUNT,
        parallel_trial_count=PARALLEL_TRIAL_COUNT,
    )

    hpt_job.run(sync=True)
    print(hpt_job.resource_name)


if __name__ == "__main__":
    submit_job()
