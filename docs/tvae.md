<div align="center">
    <h1 align="center">Video VAE Benchmark and Evaluation</h1>
</div>


## Overview

Goal: As there are many temporal VAE models available, our target is find the best baseline video VAE model as our baseline.

We select the following popular open-sourced video VAEs as candidates:

- **Hunyuan Temporal VAE**, [Code repo](https://github.com/Tencent/HunyuanVideo), [Paper link](https://arxiv.org/abs/2412.03603)
- **CosMos-Tokenizer**, [Code repo](https://github.com/NVIDIA/Cosmos-Tokenizer)
- **CogVideoX**, [Code repo](https://github.com/THUDM/CogVideo), [Paper link](https://arxiv.org/abs/2408.06072)
- **LTX-V**, [Code repo](https://github.com/Lightricks/LTX-Video/tree/main)
- **Mochi-1**, [Code repo](https://github.com/genmoai/mochi)

## Benchmark

To benchmark the performance of above baselines, following [ToeknBench](https://github.com/NVlabs/TokenBench),
we first select four video datasets suggested by TokenBench: `BDD100K`, `Bridgedata_v2`, `Ego-exo-4d`, `Panda70M`. 
Additionally, we select four popular video datasets commonly used in 3D or 4D: `DynamicReplica`, `ObjaverseXL`, 
`PointOdyssey`, `TartanAir`, `RealEstate10K`.
And we add two image datasets: `ImageNet`, `TextOCR`.

We totally select ~1k videos and 2k images from them. We provide the uids of selected videos and images in [image_bench.txt](tvae/image_bench.txt) and [video_bench.txt](tvae/video_bench.txt). We have constructed it and shared it on [Google Drive](https://drive.google.com/file/d/1bJz-dvrp1tVAAUuZuLzJbjU3ixJj1d6D/view?usp=sharing) for reproducible benchmarking.

### Image Benchmarks

#### 1. TextOCR

[TextOCR](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset) is an image dataset which contains many images with text. We sample 1k images from it.

Examples:

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_image/text_ocr/example01.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/text_ocr/example02.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/text_ocr/example05.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/text_ocr/example07.png" width="200px">
    </td>
  </tr>
</table>

#### 2. ImageNet

[ImageNet](https://www.kaggle.com/datasets/titericz/imagenet1k-val) is an image dataset contains ~1k classses. We sample 1k images from it.

Examples:

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_image/imagenet/example01.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/imagenet/example05.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/imagenet/example03.png" width="200px">
    </td>
    <td>
      <img src="tvae/example_image/imagenet/example04.png" width="200px">
    </td>
  </tr>
</table>

### Video Benchmarks

#### 1. BDD100K

[BDD100k](https://doc.bdd100k.com/download.html) is a self-driving video dataset. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/BDD100K/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/BDD100K/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/BDD100K/example03.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/BDD100K/example04.gif" width="300px">
    </td>
  </tr>
</table>

#### 2. Bridgedata_v2

[Bridgedata_v2](https://rail-berkeley.github.io/bridgedata/) is a large-scale robot learning dataset. We sample 100 videos from robot manipulation videos.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/bridgedata_v2/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/bridgedata_v2/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/bridgedata_v2/example03.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/bridgedata_v2/example04.gif" width="300px">
    </td>
  </tr>
</table>

#### 3. Ego-exo-4d

[Ego-exo-4d](https://ego-exo4d-data.org/) is a large-scale video dataset which contains both ego-centric and third-person-view videos. We sample 100 egocentric videos and 100 third-person-view videos.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/ego-exo-4d/example06.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/ego-exo-4d/example07.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/ego-exo-4d/example04.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/ego-exo-4d/example05.gif" width="300px">
    </td>
  </tr>
</table>

#### 4. Panda-70M

[Panda-70M](https://snap-research.github.io/Panda-70M/) is a large-scale Internet video dataset consisting of many diverse YouTube videos. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/panda_70m/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/panda_70m/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/panda_70m/example05.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/panda_70m/example04.gif" width="300px">
    </td>
  </tr>
</table>

#### 5. ObjaverseXL

[ObjaverseXL](https://github.com/allenai/objaverse-xl) is the largest 3D object dataset. We select 100 objects from it, render them with large camera motion.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/objaverse/example07.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/objaverse/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/objaverse/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/objaverse/example06.gif" width="300px">
    </td>
  </tr>
</table>

#### 6. PointOdyssey

[PointOdyssey](https://pointodyssey.com/) is a synthetic dataset widely used in 3D point tracking; its videos have large scene and camera motion. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/point_odyssey/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/point_odyssey/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/point_odyssey/example08.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/point_odyssey/example04.gif" width="300px">
    </td>
  </tr>
</table>

#### 7. TartanAir

[TartanAir](https://theairlab.org/tartanair-dataset/) is the largest synthetic SLAM dataset with various static scenes and environments, large drone-fly camera motion, widely used in 3D localization and reconstruction. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/tartanair/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/tartanair/example05.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/tartanair/example06.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/tartanair/example07.gif" width="300px">
    </td>
  </tr>
</table>

#### 8. RealEstate10K

[RealEstate10K](https://google.github.io/realestate10k/) is an Internet-scale video dataset consisting of many static house scans with camera motion, widely used in 3D reconstruction. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/real10k/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/real10k/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/real10k/example03.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/real10k/example04.gif" width="300px">
    </td>
  </tr>
</table>

#### 9. DynamicReplica

[DynamicReplica](https://github.com/facebookresearch/dynamic_stereo) is a synthetic dataset which contains both object and camera motion, which is usually used in dynamic 3D reconstruction. We sample 100 videos from it.

Examples:

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <img src="tvae/example_video/dynamic_replica/example01.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/dynamic_replica/example02.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/dynamic_replica/example03.gif" width="300px">
    </td>
    <td>
      <img src="tvae/example_video/dynamic_replica/example04.gif" width="300px">
    </td>
  </tr>
</table>

## Evaluation

We are going to benchmark the video VAEs with the following selected metrics:

- PSNR (Video & Image)
- SSIM (Video & Image)
- LPIPS (Video & Image)
- FLOLPIPS (Video)

:pushpin: 
- [ ] TBD: Evaluation and presentation of the results.