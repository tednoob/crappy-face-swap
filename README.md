# crappy-face-swap
No original content or contribution.

This code is heavily inspired by [FaceFusion](https://github.com/facefusion/facefusion) and it is completely powered by [InsightFace](https://github.com/deepinsight/insightface).
Do not use this tool, use FaceFusion instead, it is actively maintained, has a cool GUI, and has a good community.

Crappy in this context refers to my code alone, and is no reflection upon the projects I use, it is simply setting the correct expectations. 

# Background
I saw that FaceFusion can work with video, and it does that by converting all frames to images, and then applies magic to the images.

I wanted to try to:
* Use ffmpeg to get raw frames from the video stream without having to dump to disk in-between.
* Use less and faster magic to try and cut and fill the frame based on what InsightFace crops.
* Use less compute to go towards higher frame rates.
* Be able to run on a webcam feed, though this does not seem feasable with this approach.

# Installation
No. This barely(and rarely) runs on my machine, and I don't think it is worth the effort to get it to run.

# License
My code falls under MIT License but the models you'll need from InsightFace are as they say "non-commercial research purposes only" which means this code is worthless on its own, except as a tool for understanding.
