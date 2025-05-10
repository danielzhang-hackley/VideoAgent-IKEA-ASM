<h1>VideoAgent for IKEA ASM at Unhelkar Lab</h1>

This repo should be located installed in the remote server containing the GPU. The primary purpose of this code is to first set up a VideoLLaVA server using `video-llava.py`, and then run VQA on some videos that the user uploads to this machine using `main.py`.

<h2>File Structure</h2>

<h3>Top-level directories</h3>

The repo generally requires the following file structure.
```
annotations (all contents downloaded from IKEA ASM)
├─ Final_Annotations_Segmentation_Tracking -- contains object tracking
│  annotations for all IKEA ASM videos (not included in repo).
├─ gt_action.npy -- contains per-frame action annotations.
├─ gt_segment.npy -- annotates actions by time segments, not frames.
└─ ikea_annotation_db_full -- sqlite database for action annotation.

cache_dir
└─ Model weights for VideoLLaVA, can be downloaded from VideoAgent (not 
   included in repo).

config
└─ Basic settings for VideoAgent's VQA, the most notable being the OpenAI API
   key. More info about this under "OpenAI API Keys."

imgs
└─ Figures from VideoAgent.

installation -- more info about this under "Virtual Environments."
├─ environment.yml -- the basic conda dependencies for VideoAgent.
├─ pyproject.toml -- dependencies for VideoLLaVA.
├─ requirements.txt -- pip dependencies for VideoAgent.
├─ videoagent_final.yml -- complete conda environment for VideoAgent.
└─ videollava_final.yml -- complete conda environment for VideoLLaVA.

InternVid
└─ Module used for create the ViClip video features in temporal memory for
   VideoAgent.

LaViLa
└─ Module used to create the video segment captions in temporal memory for
   VideoAgent.

misc
└─ qa_archive.json -- A set of questions and answers for a few videos in the 
   IKEA ASM dataset.

preprocess
└─ Storage for VideoAgent's temporal and object memory.

prompts
└─ Prompts for ChatGPT for VideoAgent's tool querying system.

sample_videos
└─ A collection of videos to do VQA on. I had a subdirectory called "videos"
   within "sample_videos" containing a few IKEA ASM dataset videos.

tool_models
└─ Tool models for VQA has discussed in VideoAgent. This can be downloaded from
   the VideoAgent repo (not included in this repo).
```

<h3>Top-level files</h3>

Only need to use `main.py`, `video-llava.py`, and `evaluate_agent.py`. The former two are used for inference, and `evaluate_agent.py` is used for evaluation. More details are given below.

<h2>Virtual Environments</h2>

There are some weird behaviors with the packages in the VideoAgent repo. Below is the most consistent setup I could make. Requires Ubuntu 24.04, CUDA 12.6. For any inconsistencies, check `installations/videoagent_final.yml` and `installations/videollava_final.yml`. You can definitely try installing directly from this yaml files but I've run into issues in the past attempting this.

Create the VideoAgent environment:
```
conda env create -f installation/environment.yml
conda activate videoagent
pip install installation/requirements.txt
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

Create the Video-LLaVA environment:
```
conda create -n videollava python=3.10
cibda activate videollava
pip install --upgrade pip
cd installation
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

To debug installation issues, compare your `conda list` output against `installations/videoagent_final.yml` and `installations/videollava_final.yml`.


<h2>OpenAI API Keys</h2>

You will need an OpenAI API key to run VideoAgent. Insert it in:
- `config/default.yml`, in the `openai_api_key` field.
- `~/.bashrc`, using `export OPENAI_API_KEY=your-api-key`.


<h2>Inference</h2>

Instructions are in the <a href="https://github.com/unhelkar/comp-390-dfz">other GitHub Repo</a>, which should be installed in an individual user's local machine.


<h2>Evaluation</h2>

The functions to evaluate VideoAgent's memory are in `evaluate_videoagent.py`. These cannot be called by the scripts in the other repo, but scripts can be easily changed to do so.

Evaluation will work only for videos that are defined to be in the test set for segmentation tracking, as only these label parts. See the `segmentation_tracking_annotation.zip` file in the <a href='https://drive.google.com/drive/u/0/folders/1BKkQ7ngWcFXf_HKQpMtb2gzBag9to3tF'>IKEA ASM "annotations" folder</a>.