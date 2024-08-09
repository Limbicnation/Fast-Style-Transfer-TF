import functools
import os
import argparse

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Print TensorFlow and GPU information
print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Define image loading and visualization functions
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)

def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
    img = img[tf.newaxis, :]
    return img

def save_image(image, filename):
    """Save the image to a file."""
    if not os.path.exists('./output'):
        os.makedirs('./output')
    image = tensor_to_image(image)
    output_path = os.path.join('./output', filename)
    image.save(output_path)
    print(f"Image saved to {output_path}")

def resize_image(image, target_height):
    """Resizes the image to the target height while maintaining aspect ratio."""
    img_height, img_width = image.shape[1:3]
    scale = target_height / img_height
    new_width = int(img_width * scale)
    resized_image = tf.image.resize(image, [target_height, new_width], preserve_aspect_ratio=True)
    return resized_image

def show_image(image, title=None):
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Load the style transfer model from TensorFlow Hub
def load_hub_model():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    return hub.load(hub_handle)

# Perform style transfer using TensorFlow Hub
def perform_style_transfer(content_image_path, style_image_path, output_image_size=384, target_height=1024):
    content_image = load_image(content_image_path, image_size=(output_image_size, output_image_size))
    style_image = load_image(style_image_path, image_size=(256, 256))

    # Load the model
    hub_module = load_hub_model()

    # Perform the style transfer
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

    # Resize the stylized image to the target height while maintaining aspect ratio
    stylized_image = resize_image(stylized_image, target_height)

    # Save and visualize the stylized image
    save_image(stylized_image, 'stylized_image.jpg')
    show_image(tensor_to_image(stylized_image), 'Stylized Image')

# VGG19-based content and style extraction
def load_vgg_model():
    """Load VGG19 model without the classification head."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg

def get_content_style_representations(vgg, content_image):
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    model_outputs = content_outputs + style_outputs

    # Create a model that will return these outputs, given the model input
    model = tf.keras.models.Model(vgg.input, model_outputs)

    # Preprocess the image for VGG19
    processed_image = tf.keras.applications.vgg19.preprocess_input(content_image*255.0)

    # Get the content and style features
    outputs = model(processed_image)
    content_features = outputs[:num_content_layers]
    style_features = outputs[num_content_layers:]

    return content_features, style_features

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Perform style transfer on images.")
    parser.add_argument('--input-image', type=str, required=True, help="Path to the content image.")
    parser.add_argument('--style-image', type=str, required=True, help="Path to the style image.")
    parser.add_argument('--output-size', type=int, default=384, help="Size of the output image (default: 384).")
    parser.add_argument('--target-height', type=int, default=1024, help="Height of the final output image (default: 1024).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Perform high-level style transfer
    perform_style_transfer(args.input_image, args.style_image, args.output_size, args.target_height)

    # Load and preprocess the content image for VGG19
    content_image = load_image(args.input_image, image_size=(224, 224))

    # Load VGG19 model
    vgg_model = load_vgg_model()

    # Extract content and style representations
    content_features, style_features = get_content_style_representations(vgg_model, content_image)

    print("Content Features Shape:", [cf.shape for cf in content_features])
    print("Style Features Shape:", [sf.shape for sf in style_features])
