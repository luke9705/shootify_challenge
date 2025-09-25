# Grounded-SAM2 + Flux Kontext LoRA for On-Model Color Correction

This is a pipeline for color-accurate on-model imagery: 
1) Grounded-SAM2 automatically detects and segments the target garment; 
2) Flux Kontext-LoRA trained on side-by-side pairs — [left] reference garment (still-life) | [right] on-model — transfers the reference color while preserving texture and limiting unwanted edits.

## The process

### Dataset creation

I manually selected 60 still-life and 60 on-model high-quality images online. After having them renamed, I tried SAM2 [1] segmentation models to automatically distinguish and mask different garment weared. An example below: 

![Example of SAM2 segmentation](example.png)

The challenge with this approach lies in ensuring that we select the appropriate mask for the on-model images. I opted for Grounding DINO [2], an open-vocabulary, zero-shot object detection model that utilizes text prompts to locate and draw bounding boxes around objects in an image. This approach enabled me to detect the garment before segmenting. The outcome of this approach is essentially the Grounded-SAM2 approach [3].

However, I’ve encountered a challenge: since it’s an open-vocabulary model, we can simply label any object we want, and it returns multiple bounding boxes, ordered by decreasing accuracy. The issue is, how can we determine beforehand if the garment we want is the one with the highest accuracy? To address this, I decided to:
1) First, run Grounding DINO on the still-life (of course they have the same  number of the corresponding on-model images) to know which kind of labels the model assigned to the object, given a fixed set of labels:

```python
# Define the set of garment labels to use with Grounding DINO
labels = [
    "shirt",
    "trousers",
    "dress",
    "jacket",
    "gilet",
    "pants"
]
```

2) After the label detection, I saved the results as a dictionary in *"garments.json"*:

```json
{
    "still-life32.jpg": "jacket",
    "still-life26.jpg": "pants",
    "still-life14.jpg": "dress",
    "still-life07.jpg": "shirt",
    "still-life45.jpg": "gilet",
    "still-life53.jpg": "trousers"
    // ... more entries
}
```

3) Finally, I used the dictionary entries in Grounded-SAM2 to automatically mask the correct garment we want to degrade. Once I obtained the masks, I have just applied a random color/hue shift.

This approach resulted in a complete automatic detection, segmentation and masking process.

### Training Flux.1-Kontext

I decided to fine-tune a Diffusion Model for color-garment translation. Due to limited compute resources, I opted for training a LoRA to maximize efficiency and speed. LoRA is often sufficient to adapt big LLMs or Diffusion Models to new tasks since they already possess a substantial pre-trained knowledge base.

A traditional image-to-image pipeline with inpainting wouldn’t work for this task because it typically introduces noise in the region we want to regenerate. In this case, it would disrupt all the garment textures and shades. The only solution would be to attach a Control-Net or IP-Adapter to inject image positions and textures at each forward block to preserve the original material.

However, with the recent surge of in-image editing models like Flux.1-Kontext, Qwen-Image, GPT-Image 1, and so on, I wanted to experiment with one of these models. I chose Kontext.

The strength of these models lies in their ability to concatenate a reference image with the target latent image, allowing the model to refer to it during denoising. This process is enabled by the use of 3DRoPE, which aligns the corresponding position between the target and reference images.

From an architectural perspective, these models seem highly suitable for this kind of task; however, the only challenge is that they can only provide one reference image at a time.

While browsing the internet, I discovered that some users attempted to concatenate the images side by side, and in some cases, the model worked to make reference to one of the images while altering the other.

Therefore, I stitched the dataset images in the following manner:

![Example of SAM2 segmentation](side_side.jpg)

Finally, I reused *"garments.json"* to build the corresponding text prompt like below:

```python
t.write(f"chromiq Change the {v} color on the right to match the left")  # trigger word is "chromiq"
```

#### Training setup

To train Kontext, I used AIToolkit by Ostris, which is a well maintained LoRA Trainer for several Diffusion Models. Since the aim of the task is to only modify the right side of the images (the garment), I use the masking on the loss function in order to not waste model capability on recreating also the left side: in this way, the gradient signal was focused only on the part of the images that matters for this problem, speeding up training and maintaining efficiency. 

In addition, weight decay was set to 0 and the learning rate at 0.0001 to check as fast as possible if the approach worked.
Due to limited compute resources, I couldn't train for more than 5200 steps, taking an overall 15 hours on an Nvidia RTX 6000 Pro. For the detailed hyperparameters setup, please check *"config.yaml"*.

The problem I have encountered is that different website has different aspect ratio for the on-model images, so it was hard to resize without any strech or crop. In the end, I decided to resize and pad all the pair images with 1328 x 800, which is one of the supported Kontext resolutions.

### Results and evaluation metrics

The results are fair, considering the limited dataset and not optimal resolution for all the test images. I computed inference on the test set, which is composed by the 5 Shootify images + 3 I obtained online. Of course, the images were not used in training to ensure no data leakage and maintain the evaluation correct.

Below the qualitative results on the 8 generated images (some images are padded before and it is not the results of a poor generation):


<table>
    <tr>
        <th>Input Pair</th>
        <th>Generated Output</th>
    </tr>
    <tr>
        <td><img src="test/control/pair1.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair1.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair2.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair2.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair3.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair3.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair4.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair4.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair5.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair5.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair6.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair6.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair7.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair7.jpg" alt="Generated2" width="400"/></td>
    </tr>
        <tr>
        <td><img src="test/control/pair8.jpg" alt="Generated1" width="400"/></td>
        <td><img src="test/generated/pair8.jpg" alt="Generated2" width="400"/></td>
    </tr>
</table>



## References

[1] https://github.com/facebookresearch/sam2

[2] https://github.com/IDEA-Research/GroundingDINO

[3] https://github.com/IDEA-Research/Grounded-Segment-Anything
