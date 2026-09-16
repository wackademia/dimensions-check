# Run Size Check on a Windows PC with a GTX 1050 Ti

This guide covers a fresh clone on **64-bit Windows 10/11**, using the GTX 1050 Ti
(normally 4 GB VRAM). All paths are relative to the cloned folder. You do not need
to edit Python files or copy the old PC's virtual environment.

## 1. Install the prerequisites on the new PC

1. Install a current **NVIDIA driver that explicitly supports your GTX 1050 Ti and
   Windows version** from [NVIDIA's driver search](https://www.nvidia.com/Download/Find.aspx/).
   Restart Windows. For this CUDA 11.8 setup, use a compatible driver at least as
   new as 522.06 (the CUDA 11.8 Windows toolkit driver version).
2. Install **64-bit Python 3.11** including the Python launcher (`py`) from
   [Python for Windows](https://www.python.org/downloads/windows/). Python 3.12 is
   also supported by the setup script. Do not use Python 3.13+ for these pins.
3. Install [Git for Windows](https://git-scm.com/downloads/win).
4. Allow roughly **10 GB free disk space** for the environment, download cache,
   and models. **16 GB system RAM is recommended**; the GPU mode still holds
   model weights and image processing data in system RAM.

You do **not** need to install the standalone CUDA Toolkit, cuDNN, Visual Studio,
or compile PyTorch. The prebuilt PyTorch wheels include their CUDA runtime.
The NVIDIA graphics driver is still required.

Open a new PowerShell window and check:

```powershell
nvidia-smi
py -3.11 --version
git --version
```

`nvidia-smi` should list the GTX 1050 Ti. Its displayed CUDA version describes the
driver's supported CUDA level; it need not say 11.8. PyTorch will use its bundled
11.8 runtime.

## 2. Clone and run the setup

```powershell
git clone https://github.com/wackademia/size-check.git
cd size-check
.\setup_gtx1050ti.bat
```

If Python 3.12 is installed instead of 3.11:

```powershell
.\setup_gtx1050ti.bat -PythonVersion 3.12
```

The setup creates `.venv`, installs **PyTorch 2.7.1+cu118 and torchvision
0.22.1+cu118**, installs this repository and its constrained dependencies, downloads
the three required model files, verifies their SHA-256 checksums, and runs GPU
kernel checks. Downloads can take several minutes. Keep the window open until it
says `Setup complete`.

The batch file permits this local PowerShell setup script for that process only;
it does not permanently change the Windows execution policy. If your organization
blocks scripts, use the manual commands below under its normal policies.

The three model files total about **722 MB** and are deliberately excluded from Git:

```text
models/
  tiny_vit_inditex_finetuned.pt
  hrnet.pth
  birefnet/
    model.safetensors
```

You may copy that `models` folder from the old PC before setup. Correct files are
reused after checksum verification. Do not copy `.venv`; it contains PC-specific
interpreter paths. No model training, Hugging Face login, or paid service is needed.

## 3. Verify the GPU and the full pipeline

Run with other measurement servers stopped, so they do not compete for memory:

```powershell
.\.venv\Scripts\python.exe scripts/check_runtime.py --require-cuda --smoke
```

Expected output includes the GTX 1050 Ti, CUDA runtime `11.8`, capability `6.1`,
`automatic low-VRAM mode: True`, three valid checksums, and HTTP 200 results from
both the static and phone measurement handlers. The smoke test uses the included
sample image and an arbitrary scale: it checks execution, not physical accuracy.
It exits with an error if inference had to fall back to CPU.

This repository was developed on a CPU-only PC. The real GPU/driver check on your
GTX 1050 Ti is still required; no GPU speed or accuracy benchmark is claimed.
Local verification includes device/offloading/fallback unit tests, actual static
and phone measurement requests at 512-pixel segmentation, and a pip dry-run that
resolved the CUDA 11.8 profile for Windows Python 3.12. The local inference tests
used the existing CPU environment, not an installed CUDA 11.8 environment.

## 4. Start the static picture-upload page

```powershell
.\run_static_demo.bat
```

Wait for `Running on http://127.0.0.1:8080`, then open
**http://127.0.0.1:8080** in Chrome or Edge on that PC. Leave the terminal running.
Press **Ctrl+C** in that terminal to stop the server.

Upload the top-down image of the garment and ruler. Select the garment type and
size, click **Calibrate**, click the **0-inch mark** and **12-inch mark** on the
ruler, then measure. Use the markings, since a ruler's metal ends may extend past
them. Keep the ruler and garment in the same plane. Measurements remain in inches.
The existing M/L/XL/2XL underwear references are included in the clone; nothing
needs to be re-entered. Underwear still uses the shorts landmark model.

## 5. Start the live camera interface instead

Stop the static page first. For the camera attached to the new PC:

```powershell
.\run_phone_pc.bat
```

Open **http://127.0.0.1:8080**, allow camera access, and calibrate with the ruler.
This shortcut uses localhost HTTP, which avoids the earlier local HTTPS issue.
The camera page supports automatic underwear size estimation for **paused**
production-line items. After calibration, set the inspection area, remove the
ruler and capture the empty area, then click Start inspection. It submits one
stable snapshot per item and compares the four dimensions with your M/L/XL/2XL
references. See [the production-line setup guide](LIVE_PRODUCTION_LINE.md).
Hold each item until its result appears and leave an empty gap between items;
inference on this GPU is not promised to run at video frame rate.

For an actual phone on the same LAN, stop the PC server and run:

```powershell
.\run_phone_server.bat
ipconfig
```

Open `https://<new-PC-LAN-IPv4>:8443` on the phone, use HTTPS, proceed through the
local self-signed certificate warning, and permit camera access. Use the new PC's
Wi-Fi/Ethernet IPv4 from `ipconfig`, not the old PC's address or `127.0.0.1`.
If Windows asks, allow Python on your trusted private network. Localhost HTTP
does not give a remote phone a secure camera context. The LAN demo uses an ad-hoc
certificate; do not expose this development server on the public internet.

The separate OpenCV camera-window launcher also supports the GPU options:

```powershell
.\run_live.bat --device cuda --every 10
```

That launcher has no clickable ruler calibration UI. Supply its
`--pixels-per-unit <calculated-pixels-per-inch> --unit in` options for physical
units, or use the browser camera interface above. Press Q/Escape to close it.

## GPU behavior and options

The application tests an actual CUDA convolution, not just GPU availability.
`auto` selects a working GPU or warns and uses CPU. GPUs with at most 4.5 GiB
automatically use low-VRAM mode: only the model for the current stage is on GPU,
with the other weights in system RAM. Inference uses FP32. If a CUDA stage runs
out of memory, it retries on CPU and stays there until restart.

On this profile BiRefNet processes a **512 x 512** internal image instead of
1024 x 1024. The mask is resized back to the original image pixel grid, so the
ruler calibration math is unchanged. Mask edges and landmark estimates can change
at lower resolution; compare results with a physical tape measure before relying
on tolerances. A larger segmentation size requires more VRAM.

```powershell
# Require CUDA at startup (fail with an explanation if unavailable):
.\run_static_demo.bat --device cuda

# Smaller mask if another application is consuming VRAM:
.\run_static_demo.bat --device cuda --low-vram --segmentation-size 256

# Higher mask resolution; can exhaust 4 GB VRAM and fall back to CPU:
.\run_static_demo.bat --device cuda --low-vram --segmentation-size 1024

# Reliable CPU option, at the original mask resolution:
.\run_static_demo.bat --device cpu --segmentation-size 1024

# Different port if 8080 is already occupied:
.\run_static_demo.bat --port 8081
```

The same `--device`, `--low-vram`, and `--segmentation-size` options work with
`run_phone_pc.bat`, `run_phone_server.bat`, and `run_live.bat`. Run only one model
server at a time on the 1050 Ti. Model offloading reduces VRAM use but adds transfer
overhead; speed depends on the PC, uploaded image, and requested mask resolution.

## Manual installation (alternative to the setup script)

Run in the cloned folder, after installing the prerequisites:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-gtx1050ti.txt -c requirements-gtx1050ti.txt -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/check_runtime.py --require-cuda --smoke
```

Activation is optional: the commands already use the correct interpreter.

## Troubleshooting and updates

| Symptom | Action |
| --- | --- |
| Python executable from the old PC cannot be found | Stop servers, rename the copied `.venv` to `.venv-old`, and run setup to create a fresh environment. |
| `No matching distribution` for torch | Use 64-bit Python 3.11 or 3.12, update pip, and use the CUDA 11.8 index shown above. |
| `CUDA unavailable` / PyTorch version ends in `+cpu` | Re-run setup; check `nvidia-smi` and the driver, then run `check_runtime.py --require-cuda`. |
| `no kernel image` / unsupported GPU architecture | Reinstall the exact cu118 torch/torchvision wheels above. A newer CUDA wheel may exclude Pascal. |
| DLL import error for PyTorch | Install/repair Microsoft's x64 Visual C++ 2015–2022 Redistributable, reboot, and re-run the check. |
| Missing models or checksum error | Run `scripts/download_models.py` again; partial downloads can resume and bad complete files are downloaded again. |
| GPU memory exhausted / unexpectedly slow CPU retry | Stop other model servers/GPU apps, restart with `--low-vram --segmentation-size 256`, and re-run the smoke test. |
| Site will not open | Keep the terminal open, check it for a traceback, and use the exact printed HTTP/HTTPS address and port. |
| Port 8080 already in use | Stop the previous server with Ctrl+C or use `--port 8081`; do not start both demos on the same port. |

To update later, stop the server, run `git pull`, then re-run `setup_gtx1050ti.bat`.
Use the constraints file for any dependency reinstall. An unrestricted
`pip install --upgrade torch torchvision` can replace the Pascal-compatible build.
Keep the original local environment unchanged if testing another dependency stack;
use a separate virtual environment.

## Why these versions?

The [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271)
provide a Windows CUDA 11.8 build for torch 2.7.1 and torchvision 0.22.1.
[PyTorch's architecture-support notice](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)
documents removal of Pascal support from newer CUDA 12.8/12.9 wheels.
[NVIDIA's GTX 1050 Ti specifications](https://www.nvidia.com/content/apac/event/au/overwatch-campaign/geforce-gtx-1050.html)
identify its Pascal architecture and 4 GB memory.
[CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/)
list the Windows toolkit driver version used as the setup baseline.
