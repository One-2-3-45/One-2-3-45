## Tuning Tips:

1. The multi-view prediction module (Zero123) operates probabilistically. If some of the predicted views are not satisfactory, you may select and regenerate them.

2. In “advanced options”, you can tune two parameters as in other common diffusion models:
  - Diffusion Guidance Scale determines how much you want the model to respect the input information (input image + viewpoints). Increasing the scale typically results in better adherence, less diversity, and also higher image distortion.
  
  - Number of diffusion inference steps controls the number of diffusion steps applied to generate each image. Generally, a higher value yields better results but with diminishing returns.

Enjoy creating your 3D asset!