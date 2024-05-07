# crappy-face-swap

No original content or contribution.

This code is heavily inspired by [FaceFusion](https://github.com/facefusion/facefusion) and it is completely powered by [InsightFace](https://github.com/deepinsight/insightface).
Do not use this tool, use FaceFusion instead, it is actively maintained, has a cool GUI, and has a good community.

The camera code is taken from [Nakul Lakhotia](https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask) who in turn attributes [Miguel Grinberg](https://github.com/miguelgrinberg/flask-video-streaming).

Crappy in this context refers to my code alone, and is no reflection upon the projects I use, it is simply setting the correct expectations.

# Background

I saw that FaceFusion can work with video, and it does that by converting all frames to images, and then applies magic to the images.

I wanted to try to:

- Use ffmpeg to get raw frames from the video stream without having to dump to disk in-between.
- Use less and faster magic to try and cut and fill the frame based on what InsightFace crops.
- Use less compute to go towards higher frame rates.
- Be able to run on a webcam feed, though this does not seem feasable with this approach.

# Installation

No. This barely(and rarely) runs on my machine, and I don't think it is worth the effort to get it to run.

# Models

When I run code I do not really care, but friends do not let friends run models with arbitrary code execution. For that reason I have tried what I can to run directly with onnx-model files. I no longer remember where I found the model files I use. The large buffalo network comes directly from [InsightFace](https://github.com/deepinsight/insightface/releases/). They expose this as insightface.utils.storage.BASE_REPO_URL, but I do not like automatic downloads.
You can build GFPGANv1.4.onnx from the official pth source using [xuanandsix's tool](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo), if you do this care that the input/output names will be different from what crappy-face-swap expects.

```
SHA256
6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73  models/GFPGANv1.4.onnx
e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af  models/inswapper_128.onnx
df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc  models/buffalo_l/1k3d68.onnx
f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf  models/buffalo_l/2d106det.onnx
5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91  models/buffalo_l/det_10g.onnx
4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb  models/buffalo_l/genderage.onnx
4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  models/buffalo_l/w600k_r50.onnx
```

# License

My code falls under MIT License but the models you'll need from InsightFace are as they say "non-commercial research purposes only" which means this code is worthless on its own, except as a tool for understanding.
