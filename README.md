# ToriiGate - Batch
This tool utilizes the ToriiGate model to automatically caption multiple image files in an input-folder, for ease of process.

The model is very strong, and a fine-tune based on [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3).

It's capable of understanding concepts and images that would be impossible for most models, using the [Input Tags](https://github.com/MNeMoNiCuZ/ToriiGate-batch/tree/main?tab=readme-ov-file#input-tags) system.

The model handles all sorts of content, including NSFW / Adult images.

### Original models
- [Minthy/ToriiGate-v0.3](https://huggingface.co/Minthy/ToriiGate-v0.3)
- [Minthy/ToriiGate-v0.4-7B](https://huggingface.co/Minthy/ToriiGate-v0.4-7B)

# Official Model Showcase
> [!WARNING]
> Contains NSFW examples [https://rentry.co/q4pisesb](https://rentry.co/q4pisesb)

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/ToriiGate-batch`
2. (Optional) Create a virtual environment for your setup. Use python 3.10 to 3.11. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

# Requirements
- Tested on Python 3.11.
- Tested on Pytorch w. CUDA 12.1.

- At batch size 1 it uses ~14.2gb VRAM
- At batch size 8 it uses ~23.3gb VRAM

- At batch size 1 with `low_vram=True` it uses ~13.9gb VRAM

At 24gb vram, each batch takes ~45 seconds to process on batch 8, and 17 seconds on batch 1.

# Instructions
Place all images you wish to caption in the `/input/` directory and run `py batch-v03.py`.

## Configuration
This model can process images in 3 different output formats.
- JSON-like
- Detailed
- Brief

Change the type of output you want in the `description_type` in the CONFIG.

On top of this, it can also receive a text input to help guide it.

These are labeled as `input tags`, but from testing it can take in other information here as well.

## Input Tags
There's a config named `tag_extension`. It defaults to .tag, but this can be changed.
If there is a file with this extension next to an image, and if the `use_input_tags` is True, it will include this text-file along with the image as input.

## Why are input tags useful?
These can help you guide the caption model's outputs with a tailored input.
- It can help steer the model's output with categories it may not have included otherwise
- It can add new concepts, knowledge or words the model doesn't know about

This is incredibly powerful and makes the model much more versatile than any I have seen before as it really seems to listen quite well to the input.

There will of course be times when it fails to utilize it as you expected. But overall, this feature is incredibly valuable.

# Examples - With or without input
![Borken](https://github.com/user-attachments/assets/e40c3e05-c655-473f-97a0-a7cdccd82543)

Without input
> The image depicts a close-up portrait of a striking cat with a unique appearance. The cat's fur is predominantly white with distinct orange and brown patches, creating a visually captivating contrast. Its large, expressive eyes are a vivid orange, matching the color of its fur. The cat's ears are tall and pointed, with a slight pinkish hue on the inner side, adding to its charming and somewhat mischievous look. The cat's nose is small and pink, positioned between its prominent whiskers that are long and white, giving it a somewhat regal appearance.

> The cat is perched on a surface, possibly a table or a counter, with its front two paws firmly planted, while its back two paws are lifted, giving the impression that it is about to leap or is in mid-jump. This pose, combined with its intense gaze directed at the viewer, creates a sense of engagement and interaction.

> The background is somewhat blurred, with a few green leaves visible, suggesting that the cat is indoors, possibly near a window or a plant. The overall focus remains on the cat, with the background serving to highlight its striking features and dynamic pose.

> The image is taken at a close-up angle, with the cat's face and upper body filling most of the frame. The depth of field effect is used to blur the background, drawing the viewer's attention to the cat's detailed and expressive features.

> The atmosphere of the image is one of curiosity and engagement, as if the cat is challenging the viewer or inviting them to interact. The combination of the cat's unique coloration, its dynamic pose, and its intense gaze creates a captivating and memorable portrait that showcases the animal's natural charm and charisma.


With the following input:

`This creature is known as a Borken. It's vegetarian.`

> The image depicts a close-up view of a Borken, a fictional creature resembling a cat. The Borken has a unique appearance with a mix of feline and reptilian features. Its body is covered in soft, fluffy fur that transitions into scaly skin on its tail. The fur is primarily white with patches of orange and brown, giving it a distinct spotted pattern. The Borken's eyes are large and expressive, with a piercing yellow gaze directed straight at the viewer, creating an intense and captivating look. Its ears are tall and pointed, covered in fur, and its whiskers are long and prominent, adding to its feline-like appearance. The Borken's tail is long and slender, covered in scales that transition into fur at the base. It is raised and curved slightly, indicating alertness or curiosity. The Borken is standing on its hind legs, with its front paws resting on a surface, giving it a slightly humanoid stance. Its claws are visible and sharp, adding to its predatory appearance. The background is blurry, with hints of green foliage and a dark, indistinct surface, keeping the focus on the Borken. The overall atmosphere of the image is one of curiosity and intensity, as the Borken's direct gaze and alert posture engage the viewer.

## Conclusion
Notice how it incorporated the facts we gave it into the output for the most of it.

But it failed to incorporate the fact that the cat image was a vegetarian. While not perfect, this is still amazing functionality.

---
![Turtle](https://github.com/user-attachments/assets/919b11a4-4b0a-4e78-a392-2e4ff465d20d)

Without input
> The image depicts a terrifying and imposing monster, likely a creature from the Warhammer 40k universe, specifically a Tyranid. The creature is squatting on a large rock, its menacing presence dominating the frame. Its body is a mix of orange and dark grey scales, with the orange covering its torso and limbs, while the grey scales adorn its head, arms, and legs. The creature's head is adorned with a spiky, horn-like structure that adds to its fearsome appearance. Its eyes are small and beady, giving it a cold, calculating look. The creature's skin is wrinkled and textured, with visible muscle definition on its arms and torso, emphasizing its strength and power. Its hands are large and clawed, with sharp, menacing claws that seem ready to strike at any moment. The creature's posture is aggressive, with its arms resting on its knees, as if it is ready to pounce or attack at any second. The background is a blurry, greenish forest, which serves to further emphasize the creature's imposing presence and make it the clear focal point of the image.

With the following input:

`This creature is known as a Borken. It's the friendliest of pets.`

> The image depicts a terrifyingly cute creature known as a Borken. It has a bulky, reptilian body covered in thick, orange-brown scales. Its skin is wrinkled and textured, giving it a rugged, battle-worn appearance. The Borken's head is adorned with a menacing array of horns and spikes, some of which are sharp and jagged, while others are more rounded and smooth. Its eyes are small and beady, with a piercing, almost predatory gaze that seems to bore into the viewer's soul. The Borken's mouth is a wide, toothy grin, revealing rows of sharp, pointed teeth that seem to be constantly bared in a snarl. Its arms are thick and muscular, ending in sharp, clawed hands that appear ready to strike at any moment. The Borken's legs are similarly powerful and end in sharp claws, giving it a formidable and intimidating presence. Despite its fearsome appearance, the Borken seems to be squatting down, almost as if it is playfully offering itself to the viewer. The background is a blur of green foliage, suggesting that the Borken is outdoors in a natural setting. The overall atmosphere of the image is one of both danger and playfulness, as if the Borken is a cuddly yet deadly pet that is ready to unleash its full fury at any moment.

## Conclusion
This one mentions several times that the borken is a friendly pet. Both at the start and the end of the caption.

# Examples - Caption Types
![Borken](https://github.com/user-attachments/assets/e40c3e05-c655-473f-97a0-a7cdccd82543)

All different types support tags. The same tag as the above example is used.

## Brief
> A Borken, a cat-like creature with orange and white fur, stands on its hind legs, balancing on a surface. Its large, expressive eyes stare directly at the viewer, and its mouth is closed. Long, white whiskers extend from its face. Its orange and white fur covers its body, with a lighter patch on its chest. Sharp, orange claws grip the surface. A blurred background suggests an indoor setting with a hint of greenery.

## Detailed
> The image depicts a close-up view of a Borken, a fictional creature resembling a cat. The Borken has a unique appearance with a mix of feline and reptilian features. Its body is covered in soft, fluffy fur that transitions into scaly skin on its tail. The fur is primarily white with patches of orange and brown, giving it a distinct spotted pattern. The Borken's eyes are large and expressive, with a piercing yellow gaze that seems to look directly at the viewer. Its ears are tall and pointed, covered in fur, and its whiskers are long and prominent, adding to its feline-like appearance. The Borken's tail is long and slender, covered in scales that transition into fur at the base. The tail is raised and curved slightly, indicating alertness or curiosity. The Borken is standing on its hind legs, with its front paws resting on a surface, giving it a slightly upright and attentive posture. The background is blurry, with hints of green foliage and a dark, indistinct surface, keeping the focus on the Borken. The overall atmosphere of the image is one of curiosity and intrigue, as the Borken's direct gaze and poised stance invite the viewer to wonder about its thoughts and actions.

## JSON-like
```
{
  "character": "The image features a Borken, a fictional creature with a unique appearance. It has a mix of cat and raccoon-like features, with large, pointed cat ears and a fluffy raccoon tail. The Borken's fur is a blend of orange and white, giving it a striking and vibrant look. Its eyes are large and expressive, with a piercing gaze directed at the viewer. The creature's facial features include a small, black nose and prominent whiskers. Its body is perched on its hind legs, with its front paws resting on a surface, showcasing its agility and balance. The Borken's tail is bushy and positioned behind it, adding to its overall charm.",
  "background": "The background is blurry, with a few green leaves visible, suggesting that the Borken is indoors, possibly near a window or a plant. The depth of field effect makes the Borken stand out prominently against the blurred surroundings.",
  "texts": "None",
  "atmosphere": "The overall atmosphere of the image is one of curiosity and intrigue. The Borken's direct gaze and the depth of field effect create a sense of connection with the viewer, making the scene feel intimate and engaging. The lighting is soft, adding warmth and a sense of comfort to the image."
}
```

# gradio-webui.py
Launch this to use the tool with a gradio app.

This was created by [TekeshiX](https://huggingface.co/TekeshiX) at [https://huggingface.co/TekeshiX/ToriiGate-v0.3](https://huggingface.co/TekeshiX/ToriiGate-v0.3).

# Currently supported versions
- [ToriiGate-v0.3](https://huggingface.co/Minthy/ToriiGate-v0.3)

Hopefully [v0.4-7B](https://huggingface.co/Minthy/ToriiGate-v0.4-7B) will be supported later

# Future improvements
- Support low_vram / 4-bit quantization
