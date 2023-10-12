import sieve

from typing import List, Optional
from pydantic import BaseModel

metadata = sieve.Metadata(
    description="An active speaker detection model to detect 'whether the face in the screen is speaking or not?'.",
    image=sieve.Image(
        url="https://d3i71xaburhd42.cloudfront.net/a832f8978c55d6b127b70e1941604bfd3d1a06e6/2-Figure1-1.png"
    ),
    tags=["Video"],
    readme=open("README.md", "r").read()
)

@sieve.Model(
    name="talknet-asd",
    python_packages=[
        "torch>=1.6.0",
        "torchaudio>=0.6.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "scenedetect",
        "opencv-python",
        "python_speech_features",
        "torchvision",
        "ffmpeg",
        "gdown",
        "youtube-dl",
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg"
    ],
    run_commands=[
        "pip install pandas",
        "mkdir -p /root/.cache/models",
        "gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O /root/.cache/models/pretrain_TalkSet.model"
    ],
    cuda_version="11.8",
    gpu=True,
    metadata=metadata
)
class TalkNetASD:
    def __setup__(self):
        from demoTalkNet import setup
        self.s, self.DET = setup()

    def __predict__(self, video: sieve.Video, return_visualization: bool = False):
        """
        :param video: a video to process
        :param return_visualization: whether to return the visualization of the video.
        :return: if return_visualization is True, the first element of the tuple is the output of the model, and the second element is the visualization of the video. Otherwise, the first element is the output of the model.
        """
        from demoTalkNet import main
        # import subprocess
        # subprocess.run(["python", "demoTalkNet.py", "--videoName", video.path.split(".")[0]])
        def transform_out(out):
            outputs = []
            for o in out:
                outputs.append({
                    "frame_number": o['frame_number'],
                    "boxes": [b for b in o['faces']]
                })
            return outputs
            
        if return_visualization:
            out, video_path = main(self.s, self.DET, video.path, return_visualization=return_visualization)
            yield sieve.Video(path=video_path)
            yield transform_out(out)
        else:
            out = main(self.s, self.DET, video.path, return_visualization=return_visualization)
            yield transform_out(out)

if __name__ == "__main__":
    model = TalkNetASD()
    out = model(sieve.Video(path="/home/ubuntu/experiments/assets/kevin1.mp4"), return_visualization=False)
    print(list(out))