# Compile Baidu Apollo 9.0 on Apple Silicon chips

This is a guide repository for compiling the [Baidu Apollo](https://github.com/ApolloAuto/apollo) on macOS devices equipped with Apple Silicon chips.

Since Apollo 9.0 added support for the ARM64 architecture, it is now possible to attempt running Baidu Apollo on Apple Silicon devices.



Tested successfully on a MacBook with an M3 chip and 16GB of RAM, the compilation completes smoothly (although very slow). Dreamview (not plus) can be launched and log files can be played as expected using the provided commands.

If you encounter any issues during the build process, feel free to open an issue so we can discuss and troubleshoot together.

## Quickstart

### Prerequisites
Docker Desktop for Mac (**Apple Silicon version**): https://www.docker.com/products/docker-desktop/

### Steps

1. Clone the specified branch of Baidu Apollo on your mac
```bash
git clone https://github.com/lilejin322/BaiduApollo.git -b r9.0.1_applesilicon
```
> What was changed? One-click diff: https://github.com/lilejin322/BaiduApollo/compare/r9.0.1%E2%80%A6r9.0.1_applesilicon

2. Enter the project directory and pull the official Docker image for the corresponding Apollo version:
```bash
bash docker/scripts/dev_start.sh
```
> Currently, the `dev_start.sh` scripts may still attempt to pull some outdated, mountable auxiliary images such as `map_volume-sunnyvale-latest`. These images are built for the AMD64 architecture. However, they are not required to run the Apollo 9.0 arm64 base image `dev-aarch64-20.04-20231024_1054`, and do not need to be mounted. Therefore, architecture mismatch warnings during the pull process can be safely ignored.

3. Enter the Docker container via:
```bash
bash docker/scripts/dev_into.sh
```

4. You’re now inside the Docker container, compile the project:
```bash
bash apollo.sh build  # Execute inside the Docker Container!
```
> The compilation process may take up to an hour to complete. Due to performance bottlenecks (see the Notes section below), please be patient.

5. After successful compilation, launch Dreamview or Dreamview Plus:
- For Dreamview 1.0:
```bash
bash scripts/bootstrap.sh start  # Execute inside the Docker Container!
```
- For Dreamview Plus:
```bash
bash scripts/bootstrap.sh start_plus  # Execute inside the Docker Container!
```
> Open a browser on your Mac. Dreamview Plus is available at http://localhost:8888/, and Dreamview is available at http://localhost:8899/.
Do not use `https`!

6. Try to replay a record file, execute on your host mac's terminal:
``` bash
docker exec -it apollo_dev_$(whoami) bash -c "cd /apollo && bazel-bin/cyber/tools/cyber_recorder/cyber_recorder play -f test.00000 --loop"
```
> This file is the test recording file I added to this apollo branch, you can observe the replay in Dreamview http://localhost:8899/

## Notes

- This setup is not officially supported by the Apollo team. Moreover, running Linux-based Docker containers on Apple Silicon Macs differs from doing so on native Linux hosts. Specifically, because Docker images do not contain a Linux kernel, Docker on macOS creates an ARM64 Linux virtual machine (VM) to support ARM-based Linux containers. As a result, Docker containers do not run directly on macOS but instead within this VM layer, which introduces additional overhead and reduces runtime efficiency.

- Despite these limitations, this approach is a great convenience for software developers and test engineers. If you’re primarily using macOS, you can still use tools like `cyber_recorder` to replay recorded data and observe Apollo’s behavior — all without needing a separate Linux machine.
