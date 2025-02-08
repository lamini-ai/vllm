# Lamini fork of vLLM

This is for added features needed for Lamini Platform.

## Naming

For branch names, we use the following naming convention:
`lamini-<vLLM_VERSION>`

e.g. `lamini-v0.6.5`, `lamini-v0.6.6`, etc.

For image tags, we use the following naming convention:
`<vLLM_VERSION>-<LAMINI_VERSION>`

e.g. `v0.6.5-000`, `v0.6.5-001`, `v0.6.5-002`, etc.

## Updating from upstream
On the latest lamini branch, checkout one for the new version (this will keep all lamini commits)

e.g. `git checkout -b lamini-v0.7.2`

```bash
# Fetch upstream tags
$ git fetch upstream --tags

# Rebase
$ git rebase v0.7.2

# Push
git push --set-upstream origin lamini-v0.7.2
```

Finally, update the default branch in the repo settings to the latest version.
<img width="852" alt="Screenshot 2025-02-07 at 6 44 44 PM" src="https://github.com/user-attachments/assets/031cc358-b501-4f98-afd0-9d3801e9a5be" />

## Github Actions

Github Actions have been disabled for this repo to avoid triggering vLLM's workflows. They can be re-enabled in the repo settings.

## Building the Docker image

### RoCM

```bash
sudo DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t powerml/inference-engine-amd:<INSERT_TAG> .
```

<details>
<summary>Push the image to Docker Hub</summary>
If you want to push the image to Docker Hub, you can use the following commands:

```bash
docker login
docker push powerml/inference-engine-amd:<INSERT_TAG>
```

</details>

### Nvidia

Building the vLLM Docker image with Nvidia takes ~1 hour.

```bash
sudo DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag powerml/inference-engine-nvidia:<INSERT_TAG>
```

If you are running on a more powerful machine, you can increase the number of jobs and threads according to the number of CPU cores on your machine.

```bash
# To get number of CPU cores
$ lscpu
# or
$ ncpus

# Build the Docker image with more jobs and threads
$ sudo DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag powerml/inference-engine-nvidia:<INSERT_TAG> --build-arg max_jobs=<LESS_THAN_OR_EQUAL_TO_CPU_CORES> --build-arg nvcc_threads=8
```

If you run into a vLLM image size error, you can disable the wheel check with `--build-arg RUN_WHEEL_CHECK=false` and run the build command again.

<details>
<summary>Push the image to Docker Hub</summary>
If you want to push the image to Docker Hub, you can use the following commands:

```bash
docker login
docker push powerml/inference-engine-nvidia:<INSERT_TAG>
```

</details>
