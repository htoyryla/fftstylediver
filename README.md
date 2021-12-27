# fftstylediver

Start from scratch or from an init image. Let the image grow towards

1 a content image
2 the "style" of a style image
3 a text prompt

Familiarity with Gram matrix based style transfer and CLIP based image synthesis is recommended.

Use pytorch 1.7

Example

```
 python fftstylediver9d.py --image your-content-target-image  --style your-style-target-image --cuda --name folder/basename --saveEvery 10 --niter 5000 --sw 960 --imageSize 960  --cw 400 --tw 3 --lr 0.1 --text "your text prompt" --low 0.7 --high 0.95 --cutn 8 --init your-init-image --showSize 640
```

 
 Options:
 ```
 --cuda use cuda
 --sw style weight
 --cw content weight
 --tw text weight
 --show display progress in a window on local screen
 --showSize set size for the local window
 ```
 
 
 
 
