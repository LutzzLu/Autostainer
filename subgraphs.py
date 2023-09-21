import umap
import colorsys
import matplotlib.pyplot as plt

def draw_subgraph_embeddings_with_umap():
    um = umap.UMAP()
    u = um.fit_transform(subgraph_embeddings['predictions'].cpu())

    u_norm = (u - u.mean(axis=0)) / (u.std(axis=0))
    magnitude = (u_norm[:, 0] ** 2 + u_norm[:, 1] ** 2) ** 0.5
    angle = (np.arctan2(u_norm[:, 1], u_norm[:, 0]) + np.pi) / (2 * np.pi)

    # Starts at 1 and gradually gets smaller
    saturation = 1/(magnitude + 1)

    # All are 0.5
    value = np.ones_like(saturation)

    colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(angle, saturation, value)])

    plt.scatter(u[:, 0], u[:, 1], c=colors)
    plt.show()

    ds = 8

    # Drawing this back on the slide, to see where similar tissue goes

    im = slide.image[:, ::ds, ::ds].clone()

    for color, image_x, image_y in zip(colors, slide.spot_locations.image_x // ds, slide.spot_locations.image_y // ds):
        r = 96 // ds
        im[:, image_y - r:image_y + r, image_x - r:image_x + r] = torch.tensor(color)[:, None, None]

    plt.imshow((im.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
    plt.show()
    plt.imshow(slide.render(8, slide.spot_counts[:, 16], spot_size=128))
    plt.show()