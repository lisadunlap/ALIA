from dataclasses import dataclass

@dataclass
class Txt2ImgSingleArgs:
    """Text-2-Image Generation"""

    prompt: str
    """Prompt for image (e.g. a dignified beaver with glasses, digital art)."""

    save_dir: str = './diffusion_generated_data'
    """Where to save the generated images."""

    n: int = 50
    """Number of images to generate."""

    test: bool = False
    """Mainly for development."""

    wandb_silent: bool = False
    """Turn WandB logging off, but why would you want to do that?"""

    model: str = 'runwayml/stable-diffusion-v1-5'
    """Huggingface model_id"""

    safety_checker: bool = False
    """Enable safety checker"""

    mem_num_imgs: int = 2
    """Number of images that can be generated at once without an OOM error."""

@dataclass
class Img2ImgSingleArgs(Txt2ImgSingleArgs):

    im_path: str = './ex_imgs/giraffe.png'
    """Path to image you want to edit."""

    strength: float = 0.6
    """Edit strength (0-1). Higher it is the more noticable the edits are, but at the risk of distorting the class-relevant information."""

    guidance: float = 5 
    """Text guidance. Higher values produce outputs that more explicitly conform to the provided text, but at the risk of distorting the class-relevant information."""

    n: int = 2
    """Number of edits to generate per image."""


@dataclass
class Txt2ImgArgs(Txt2ImgSingleArgs):

    dataset: str = 'Cub2011'
    """Dataset to generate images for. We extract the class names from the class_names attribute."""

    data_dir: str = '/shared/lisabdunlap/data'
    """Where the dataset is stored."""

@dataclass
class Img2ImgArgs(Img2ImgSingleArgs):
    
    prompt: str = 'a photo of a {}.'
    """Prompt for image (e.g. a cameratrap photo of a {}). Use {} in place of the class name unless class_agnostic is set to True"""

    dataset: str = 'Cub2011'
    """Dataset to generate images for. We extract the class names from the class_names attribute."""

    grid_log_freq: int = 100
    """How often to log image grid."""

    class_agnostic: bool = False
    """Use class agnostic prompt."""

    data_dir: str = '/shared/lisabdunlap/data'
    """Where the dataset is stored."""

@dataclass
class InstructPix2PixSingleArgs(Img2ImgArgs):

    image_guidance: float = 1.2
    """How faithful to stay to the image (>= 1)"""

    prompt: str = 'put the {} in the snow.'
    """Prompt for image (e.g. put the {} in the wild). Use {} in place of the class name unless class_agnostic is set to True"""

    im_path: str = './ex_imgs/giraffe.png'
    """Path to image you want to edit."""

    guidance: float = 5 
    """Text guidance. Higher values produce outputs that more explicitly conform to the provided text, but at the risk of distorting the class-relevant information."""

    n: int = 2
    """Number of edits to generate per image."""

    model: str = 'timbrooks/instruct-pix2pix'
    """Huggingface model_id"""

@dataclass
class InstructPix2PixArgs(InstructPix2PixSingleArgs):

    dataset: str = 'Cub2011'
    """Dataset to generate images for. We extract the class names from the class_names attribute."""

    grid_log_freq: int = 100
    """How often to log image grid."""

    class_agnostic: bool = False
    """Use class agnostic prompt."""

    data_dir: str = '/shared/lisabdunlap/data'
    """Where the dataset is stored."""