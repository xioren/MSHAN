from utils import *
from generator import MSHAN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(input_image_path, batch_size=128, patch_size=64, upscale_factor=2, model_path="models/checkpoint.pth"):
    # Load the trained generator model
    generator = MSHAN()
    checkpoint = torch.load(model_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    generator.to(device)

    # Load and preprocess the input image
    image_tensor = load_and_transform_image(input_image_path).unsqueeze(0)

    # Chunk the image
    chunks = extract_tensor_patches(image_tensor, patch_size=patch_size)

    # Process each chunk through the generator model
    processed_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = torch.cat(chunks[i:i + batch_size]) # Move processed chunks to CPU and store them
        batch = batch.to(device)
        with torch.no_grad():
            processed_batch = generator(batch)
        processed_chunks.extend(processed_batch.cpu()) # Store the processed chunks

    # Convert processed chunks to PIL images
    processed_chunks = [tensor_to_pil(chunk, normalize=True) for chunk in processed_chunks]

    # Reassemble processed chunks
    _, _, original_height, original_width = image_tensor.shape # Extract original dimensions
    reassembled_image = recompile_pil_patches(processed_chunks, upscale_factor * original_width, upscale_factor * original_height, patch_size=patch_size*upscale_factor)

    # Create output path
    base, end, = os.path.split(input_image_path)
    fn, ext = os.path.splitext(end)
    output_image_path = os.path.join(base, f"{fn}-{upscale_factor}x.png")

    # Save the output image
    save_pil_image(reassembled_image, output_image_path)
    print(f"Saved generated image to {output_image_path}")


if __name__ == "__main__":
    inference("/path/to/image", batch_size=32)
